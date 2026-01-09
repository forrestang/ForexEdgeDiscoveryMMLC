"""
LSTM data processing module.
Handles CSV file scanning, date-range parquet creation, ADR calculation,
timeframe upsampling, and file management.
"""

import polars as pl
from pathlib import Path
from datetime import date, time
import re
from typing import Optional

from app.core.csv_parser import (
    extract_pair_from_filename,
    scan_source_folder,
    load_and_stitch_pair_data,
)
from app.core.upsampler import upsample_ohlc
from app.core.timezone import est_to_utc
from app.config import settings, SESSION_TIMES, TIMEFRAMES


def get_lstm_path(working_dir: Optional[str] = None, subfolder: str = "raw") -> Path:
    """Get the LSTM directory path, creating it if needed.

    LSTM data is stored separately from AE+KNN cache at: working_dir/lstm/{subfolder}/

    Args:
        working_dir: Base working directory
        subfolder: Subfolder within lstm/ (default: "raw")

    Returns:
        Path to lstm/{subfolder}/
    """
    base = Path(working_dir) if working_dir else settings.default_working_directory
    lstm_dir = base / "lstm" / subfolder
    lstm_dir.mkdir(parents=True, exist_ok=True)
    return lstm_dir


def get_data_path(working_dir: Optional[str] = None) -> Path:
    """Get the data directory path."""
    base = Path(working_dir) if working_dir else settings.default_working_directory
    return base / settings.data_folder_name


def extract_year_from_filename(filename: str) -> Optional[int]:
    """Extract year from DAT_MT_[PAIR]_M1_[YEAR].csv or DAT_NT_[PAIR]_M1_[YEAR].csv format."""
    match = re.search(r"_M1_(\d{4})\.csv", filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def extract_info_from_filename(filename: str) -> dict:
    """
    Extract pair, year, and file type from various filename patterns.
    Supports:
    - DAT_MT_[PAIR]_M1_[YEAR].csv / DAT_NT_[PAIR]_M1_[YEAR].csv (standard data)
    - Files containing 'test' or 'validation' (test/validation files)
    """
    filename_lower = filename.lower()

    # Check for test/validation files
    is_test = "test" in filename_lower or "validation" in filename_lower

    # Try standard pattern first
    pair = extract_pair_from_filename(filename)
    year = extract_year_from_filename(filename)

    # Determine file type
    if is_test:
        file_type = "test"
    else:
        file_type = "data"

    return {
        "pair": pair,
        "year": year,
        "type": file_type,
    }


def scan_data_files(working_dir: Optional[str] = None) -> list[dict]:
    """
    Scan the data folder for CSV files and return metadata.
    Returns list of dicts with: name, pair, year, size_mb, type
    Includes standard data files and test/validation files.
    """
    data_path = get_data_path(working_dir)
    if not data_path.exists():
        return []

    files = []
    seen_files = set()

    # Scan for standard data files
    for file_path in list(data_path.glob("DAT_MT_*_M1_*.csv")) + list(data_path.glob("DAT_NT_*_M1_*.csv")):
        if file_path.name in seen_files:
            continue
        seen_files.add(file_path.name)

        info = extract_info_from_filename(file_path.name)
        size_mb = file_path.stat().st_size / (1024 * 1024)

        files.append({
            "name": file_path.name,
            "pair": info["pair"],
            "year": info["year"],
            "type": info["type"],
            "size_mb": round(size_mb, 2),
        })

    # Scan for test/validation files (any CSV with 'test' or 'validation' in name)
    for file_path in data_path.glob("*.csv"):
        if file_path.name in seen_files:
            continue

        filename_lower = file_path.name.lower()
        if "test" in filename_lower or "validation" in filename_lower:
            seen_files.add(file_path.name)

            info = extract_info_from_filename(file_path.name)
            size_mb = file_path.stat().st_size / (1024 * 1024)

            files.append({
                "name": file_path.name,
                "pair": info["pair"],
                "year": info["year"],
                "type": info["type"],
                "size_mb": round(size_mb, 2),
            })

    # Sort by type (data first, then test), then pair, then year
    files.sort(key=lambda x: (x["type"] != "data", x["pair"] or "", x["year"] or 0))
    return files


def list_lstm_parquets(working_dir: Optional[str] = None) -> list[dict]:
    """
    List existing LSTM parquet files from the raw/ subfolder.
    Returns list of dicts with: name, pair, start_date, end_date, timeframe, adr_period, size_mb, rows

    Supports both old format (PAIR_YYYY-MM-DD_to_YYYY-MM-DD.parquet)
    and new format (PAIR_YYYY-MM-DD_to_YYYY-MM-DD_ADR20_M5.parquet)
    """
    lstm_dir = get_lstm_path(working_dir, subfolder="raw")
    parquets = []

    for file_path in lstm_dir.glob("*.parquet"):
        pair = None
        start_date = None
        end_date = None
        timeframe = None
        adr_period = None

        # Try new format: PAIR_YYYY-MM-DD_to_YYYY-MM-DD_ADR{n}_{TF}.parquet
        match = re.match(
            r"(\w+)_(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})_ADR(\d+)_(\w+)\.parquet",
            file_path.name
        )
        if match:
            pair = match.group(1)
            start_date = match.group(2)
            end_date = match.group(3)
            adr_period = int(match.group(4))
            timeframe = match.group(5)
        else:
            # Try old format: PAIR_YYYY-MM-DD_to_YYYY-MM-DD.parquet
            match = re.match(
                r"(\w+)_(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})\.parquet",
                file_path.name
            )
            if match:
                pair = match.group(1)
                start_date = match.group(2)
                end_date = match.group(3)
            else:
                # Unknown format, just use filename
                pair = file_path.stem

        size_mb = file_path.stat().st_size / (1024 * 1024)

        # Read row count from parquet file
        rows = 0
        try:
            df = pl.scan_parquet(file_path)
            rows = df.select(pl.len()).collect().item()
        except Exception:
            pass

        parquets.append({
            "name": file_path.name,
            "pair": pair,
            "start_date": start_date,
            "end_date": end_date,
            "timeframe": timeframe,
            "adr_period": adr_period,
            "size_mb": round(size_mb, 2),
            "rows": rows,
        })

    parquets.sort(key=lambda x: (x["pair"] or "", x["timeframe"] or "", x["start_date"] or "", x["name"]))
    return parquets


def calculate_session_adrs(df: pl.DataFrame, adr_period: int = 20) -> pl.DataFrame:
    """
    Calculate ADR for each session type and add as columns.

    For each session (full_day, asia, london, ny):
    1. Filter M1 bars to session time window
    2. Group by date, calc session_range = max(high) - min(low)
    3. Shift by 1 (exclude current day), rolling_mean(window_size=adr_period)
    4. Join back to main dataframe as adr_{session} column

    Args:
        df: DataFrame with M1 OHLC data (timestamp, open, high, low, close, volume)
        adr_period: Number of days for ADR lookback (default 20)

    Returns:
        DataFrame with adr_full_day, adr_asia, adr_london, adr_ny columns added,
        trimmed to remove rows without complete ADR history
    """
    # Add date column for grouping
    df = df.with_columns(
        pl.col("timestamp").dt.date().alias("trade_date")
    )

    # Calculate ADR for each session
    for session_name, ((start_h, start_m), (end_h, end_m)) in SESSION_TIMES.items():
        start_time = time(start_h, start_m)
        end_time = time(end_h, end_m)

        # Filter bars within session time window
        session_bars = df.filter(
            (pl.col("timestamp").dt.time() >= start_time) &
            (pl.col("timestamp").dt.time() < end_time)
        )

        # Calculate daily range for this session
        # Group by date, get max(high) - min(low)
        daily_ranges = (
            session_bars.group_by("trade_date")
            .agg([
                (pl.col("high").max() - pl.col("low").min()).alias("session_range")
            ])
            .sort("trade_date")
        )

        # Calculate rolling ADR (average of previous N days, excluding current day)
        daily_ranges = daily_ranges.with_columns(
            pl.col("session_range")
            .shift(1)  # Shift by 1 to exclude current day
            .rolling_mean(window_size=adr_period)
            .alias(f"adr_{session_name}")
        )

        # Join ADR back to the main dataframe
        df = df.join(
            daily_ranges.select(["trade_date", f"adr_{session_name}"]),
            on="trade_date",
            how="left"
        )

    # Drop the temporary trade_date column
    df = df.drop("trade_date")

    # Filter out rows where ANY ADR is null (first N days don't have complete history)
    # Use full_day ADR as the primary filter since it's the most comprehensive
    df = df.filter(pl.col("adr_full_day").is_not_null())

    return df


def upsample_with_adr(df: pl.DataFrame, target_timeframe: str) -> pl.DataFrame:
    """
    Upsample M1 data to higher timeframe while preserving ADR columns.

    ADR columns are carried forward using first() aggregation (same as open).

    Args:
        df: DataFrame with M1 OHLC data and ADR columns
        target_timeframe: Target timeframe (M1, M5, M10, M15, M30, H1, H4)

    Returns:
        Upsampled DataFrame with ADR columns preserved
    """
    if target_timeframe == "M1":
        return df

    from app.config import TIMEFRAME_INTERVALS
    interval = TIMEFRAME_INTERVALS[target_timeframe]

    # Build aggregation list - OHLCV + ADR columns
    agg_exprs = [
        pl.col("open").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").last().alias("close"),
        pl.col("volume").sum().alias("volume"),
    ]

    # Add ADR column aggregations (first value in group, same as open)
    for session_name in SESSION_TIMES.keys():
        adr_col = f"adr_{session_name}"
        if adr_col in df.columns:
            agg_exprs.append(pl.col(adr_col).first().alias(adr_col))

    return df.group_by_dynamic(
        "timestamp",
        every=interval,
        closed="left",
        label="left",
    ).agg(agg_exprs)


def create_date_range_parquet(
    pair: str,
    start_date: date,
    end_date: date,
    working_dir: Optional[str] = None,
) -> dict:
    """
    Create a parquet file for a specific pair and date range.

    Args:
        pair: Currency pair (e.g., "EURUSD")
        start_date: Start date for the data range
        end_date: End date for the data range
        working_dir: Working directory path

    Returns:
        Dict with status, filename, rows, size_mb
    """
    data_path = get_data_path(working_dir)
    lstm_dir = get_lstm_path(working_dir)

    # Scan for files matching this pair
    files_by_pair = scan_source_folder(data_path)

    if pair not in files_by_pair:
        return {
            "status": "error",
            "message": f"No CSV files found for pair {pair}",
            "filename": None,
            "rows": 0,
            "size_mb": 0,
        }

    # Load and stitch all files for this pair
    df = load_and_stitch_pair_data(files_by_pair[pair])

    # Convert EST timestamps to UTC for consistency with rest of system
    df = est_to_utc(df)

    if df.is_empty():
        return {
            "status": "error",
            "message": f"No data loaded for pair {pair}",
            "filename": None,
            "rows": 0,
            "size_mb": 0,
        }

    # Filter to date range
    start_dt = pl.datetime(start_date.year, start_date.month, start_date.day)
    end_dt = pl.datetime(end_date.year, end_date.month, end_date.day, 23, 59, 59)

    df_filtered = df.filter(
        (pl.col("timestamp") >= start_dt) & (pl.col("timestamp") <= end_dt)
    )

    if df_filtered.is_empty():
        return {
            "status": "error",
            "message": f"No data found in date range {start_date} to {end_date}",
            "filename": None,
            "rows": 0,
            "size_mb": 0,
        }

    # Generate filename
    filename = f"{pair}_{start_date.isoformat()}_to_{end_date.isoformat()}.parquet"
    output_path = lstm_dir / filename

    # Save to parquet with compression
    df_filtered.write_parquet(output_path, compression="zstd")

    size_mb = output_path.stat().st_size / (1024 * 1024)

    return {
        "status": "success",
        "message": f"Created {filename}",
        "filename": filename,
        "rows": len(df_filtered),
        "size_mb": round(size_mb, 2),
    }


def create_parquets_from_files(
    selected_files: list[str],
    working_dir: Optional[str] = None,
    adr_period: int = 20,
    timeframes: list[str] = None,
) -> dict:
    """
    Create parquet files from selected CSV files.
    Groups files by pair, calculates ADR, upsamples to selected timeframes,
    and creates one parquet per pair per timeframe.

    Pipeline:
    1. Load and stitch M1 CSVs, deduplicate by timestamp
    2. Calculate session ADRs on M1 data (most accurate high/low)
    3. For each selected timeframe: upsample M1 → target, save with ADR

    Args:
        selected_files: List of CSV filenames to process
        working_dir: Working directory path
        adr_period: Number of days for ADR lookback (default 20)
        timeframes: List of timeframes to create (default ["M5"])

    Returns:
        Dict with status, created parquets info, and any errors
    """
    from app.core.csv_parser import parse_metatrader_csv

    if timeframes is None:
        timeframes = ["M5"]

    data_path = get_data_path(working_dir)
    lstm_dir = get_lstm_path(working_dir, subfolder="raw")

    # Group selected files by pair
    files_by_pair: dict[str, list[Path]] = {}
    for filename in selected_files:
        file_path = data_path / filename
        if not file_path.exists():
            continue

        info = extract_info_from_filename(filename)
        pair = info["pair"]
        if not pair:
            # Try to extract pair from filename for non-standard files
            pair = filename.split("_")[0].upper() if "_" in filename else filename.split(".")[0].upper()

        if pair not in files_by_pair:
            files_by_pair[pair] = []
        files_by_pair[pair].append(file_path)

    if not files_by_pair:
        return {
            "status": "error",
            "message": "No valid files found",
            "created": [],
            "errors": [],
        }

    created = []
    errors = []

    for pair, file_paths in files_by_pair.items():
        try:
            # Sort files by name (chronological order)
            file_paths.sort(key=lambda p: p.name)

            # Load and stitch all files
            dfs = []
            for fp in file_paths:
                df = parse_metatrader_csv(fp)
                dfs.append(df)

            if not dfs:
                errors.append(f"{pair}: No data loaded")
                continue

            # Concatenate and deduplicate
            combined = pl.concat(dfs)
            combined = combined.sort("timestamp").unique(subset=["timestamp"], keep="first")

            # Convert EST timestamps to UTC for consistency with rest of system
            combined = est_to_utc(combined)

            if combined.is_empty():
                errors.append(f"{pair}: Empty after processing")
                continue

            # Get row count before ADR trimming
            rows_before_adr = len(combined)

            # Calculate ADRs on M1 data (most accurate high/low)
            # This also trims the first N days without complete ADR history
            combined_with_adr = calculate_session_adrs(combined, adr_period)

            if combined_with_adr.is_empty():
                errors.append(f"{pair}: Empty after ADR calculation - need at least {adr_period + 1} days of data")
                continue

            # Calculate trimmed days
            rows_after_adr = len(combined_with_adr)
            # Estimate trimmed days (M1 data = ~1440 bars per day for 24h or ~1320 for 22h trading)
            bars_per_day_estimate = 1320  # 22 hours × 60 minutes
            trimmed_rows = rows_before_adr - rows_after_adr
            trimmed_days = max(0, trimmed_rows // bars_per_day_estimate)

            # Get date range from actual data after trimming
            min_date = combined_with_adr["timestamp"].min()
            max_date = combined_with_adr["timestamp"].max()

            start_date_str = min_date.strftime("%Y-%m-%d")
            end_date_str = max_date.strftime("%Y-%m-%d")

            # Create parquet for each selected timeframe
            for tf in timeframes:
                try:
                    # Upsample to target timeframe with ADR columns preserved
                    upsampled = upsample_with_adr(combined_with_adr, tf)

                    if upsampled.is_empty():
                        errors.append(f"{pair}_{tf}: Empty after upsampling")
                        continue

                    # Generate filename with ADR period and timeframe
                    # Format: PAIR_YYYY-MM-DD_to_YYYY-MM-DD_ADR{n}_{TF}.parquet
                    filename = f"{pair}_{start_date_str}_to_{end_date_str}_ADR{adr_period}_{tf}.parquet"
                    output_path = lstm_dir / filename

                    # Save to parquet with compression
                    upsampled.write_parquet(output_path, compression="zstd")

                    size_mb = output_path.stat().st_size / (1024 * 1024)

                    created.append({
                        "pair": pair,
                        "filename": filename,
                        "rows": len(upsampled),
                        "size_mb": round(size_mb, 2),
                        "start_date": start_date_str,
                        "end_date": end_date_str,
                        "adr_period": adr_period,
                        "timeframe": tf,
                        "trimmed_days": trimmed_days,
                    })

                except Exception as e:
                    errors.append(f"{pair}_{tf}: {str(e)}")

        except Exception as e:
            errors.append(f"{pair}: {str(e)}")

    return {
        "status": "success" if created else "error",
        "message": f"Created {len(created)} parquet(s)" if created else "No parquets created",
        "created": created,
        "errors": errors,
    }


def delete_lstm_parquet(filename: str, working_dir: Optional[str] = None) -> dict:
    """
    Delete an LSTM parquet file from the raw/ subfolder.

    Args:
        filename: Name of the parquet file to delete
        working_dir: Working directory path

    Returns:
        Dict with status and message
    """
    lstm_dir = get_lstm_path(working_dir, subfolder="raw")
    file_path = lstm_dir / filename

    if not file_path.exists():
        return {
            "status": "error",
            "message": f"File not found: {filename}",
        }

    # Safety check - only delete .parquet files in the lstm cache folder
    if not filename.endswith(".parquet"):
        return {
            "status": "error",
            "message": "Can only delete parquet files",
        }

    file_path.unlink()

    return {
        "status": "success",
        "message": f"Deleted {filename}",
    }


def delete_lstm_parquets_batch(filenames: list[str], working_dir: Optional[str] = None) -> dict:
    """
    Delete multiple LSTM parquet files from the raw/ subfolder.

    Args:
        filenames: List of parquet file names to delete
        working_dir: Working directory path

    Returns:
        Dict with status, deleted list, and errors list
    """
    lstm_dir = get_lstm_path(working_dir, subfolder="raw")
    deleted = []
    errors = []

    for filename in filenames:
        file_path = lstm_dir / filename

        # Safety check - only delete .parquet files
        if not filename.endswith(".parquet"):
            errors.append(f"{filename}: Can only delete parquet files")
            continue

        if not file_path.exists():
            errors.append(f"{filename}: File not found")
            continue

        try:
            file_path.unlink()
            deleted.append(filename)
        except Exception as e:
            errors.append(f"{filename}: {str(e)}")

    return {
        "status": "success" if deleted else "error",
        "message": f"Deleted {len(deleted)} file(s)" if deleted else "No files deleted",
        "deleted": deleted,
        "errors": errors,
    }
