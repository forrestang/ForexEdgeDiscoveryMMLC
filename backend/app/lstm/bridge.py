"""
LSTM Bridge - Enriches raw parquet files with MMLC state and outcomes

This module coordinates the enrichment process:
1. Loads raw parquet files from lstm/raw/
2. Runs MMLC independently for 6 sessions per day
3. Outputs enriched parquet files to lstm/bridged/

The enriched files contain 78 new columns (13 per session x 6 sessions):
- State: {prefix}_state_level, {prefix}_state_dir, {prefix}_state_event
- Outcome (raw): {prefix}_out_next, {prefix}_out_next5, {prefix}_out_sess, {prefix}_out_max_up, {prefix}_out_max_down
- Outcome (normalized by ADR): {prefix}_out_next_norm, {prefix}_out_next5_norm, {prefix}_out_sess_norm,
  {prefix}_out_max_up_norm, {prefix}_out_max_down_norm
"""

import polars as pl
from pathlib import Path
from datetime import date
from typing import Optional, Callable
import time as time_module
import re

from app.lstm.data_processor import get_lstm_path
from app.lstm.session_processor import SessionProcessor, SESSION_PREFIXES

# Map session type to ADR column name
SESSION_TO_ADR_COL = {
    "asia": "adr_asia",
    "london": "adr_london",
    "ny": "adr_ny",
    "full_day": "adr_full_day",
    "asia_london": "adr_asia_london",
    "london_ny": "adr_london_ny",
}


def parse_raw_parquet_filename(filename: str) -> dict:
    """
    Parse raw parquet filename to extract metadata.

    Expected format: {PAIR}_{YYYY-MM-DD}_to_{YYYY-MM-DD}_ADR{period}_{TIMEFRAME}.parquet
    Example: EURUSD_2020-01-01_to_2024-12-31_ADR20_M5.parquet
    """
    result = {
        "pair": None,
        "start_date": None,
        "end_date": None,
        "adr_period": None,
        "timeframe": None,
    }

    # Remove .parquet extension
    name = filename.replace(".parquet", "")

    # Try to match the pattern
    pattern = r"^([A-Z]{6})_(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})_ADR(\d+)_([A-Z0-9]+)$"
    match = re.match(pattern, name)

    if match:
        result["pair"] = match.group(1)
        result["start_date"] = match.group(2)
        result["end_date"] = match.group(3)
        result["adr_period"] = int(match.group(4))
        result["timeframe"] = match.group(5)
    else:
        # Try simpler pattern without ADR
        pattern2 = r"^([A-Z]{6})_(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})$"
        match2 = re.match(pattern2, name)
        if match2:
            result["pair"] = match2.group(1)
            result["start_date"] = match2.group(2)
            result["end_date"] = match2.group(3)

    return result


def parse_bridged_parquet_filename(filename: str) -> dict:
    """
    Parse bridged parquet filename to extract metadata.

    Expected format: {PAIR}_{YYYY-MM-DD}_to_{YYYY-MM-DD}_ADR{period}_{TIMEFRAME}_bridged.parquet
    Example: EURUSD_2020-01-01_to_2024-12-31_ADR20_M5_bridged.parquet
    """
    result = {
        "pair": None,
        "start_date": None,
        "end_date": None,
        "adr_period": None,
        "timeframe": None,
    }

    # Remove _bridged.parquet suffix
    name = filename.replace("_bridged.parquet", "")

    # Try to match the full pattern with ADR and timeframe
    pattern = r"^([A-Z]{6})_(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})_ADR(\d+)_([A-Z0-9]+)$"
    match = re.match(pattern, name)

    if match:
        result["pair"] = match.group(1)
        result["start_date"] = match.group(2)
        result["end_date"] = match.group(3)
        result["adr_period"] = int(match.group(4))
        result["timeframe"] = match.group(5)
    else:
        # Try simpler pattern without ADR
        pattern2 = r"^([A-Z]{6})_(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})$"
        match2 = re.match(pattern2, name)
        if match2:
            result["pair"] = match2.group(1)
            result["start_date"] = match2.group(2)
            result["end_date"] = match2.group(3)

    return result


class LSTMBridge:
    """
    Orchestrates the LSTM bridge enrichment process.

    Takes raw parquet files and produces enriched files with:
    - 7 columns per session x 6 sessions = 42 new columns
    - Each session processed independently with fresh MMLC instance per day
    """

    def __init__(self, working_dir: Optional[str] = None):
        self.working_dir = working_dir
        self.raw_path = get_lstm_path(working_dir, subfolder="raw")
        self.bridged_path = get_lstm_path(working_dir, subfolder="bridged")

    def list_raw_parquets(self) -> list[dict]:
        """List available raw parquet files with metadata."""
        parquets = []
        for file_path in self.raw_path.glob("*.parquet"):
            size_mb = file_path.stat().st_size / (1024 * 1024)

            # Parse filename for metadata
            meta = parse_raw_parquet_filename(file_path.name)

            # Get row count
            try:
                df = pl.scan_parquet(file_path)
                rows = df.select(pl.len()).collect().item()
            except Exception:
                rows = 0

            parquets.append({
                "name": file_path.name,
                "pair": meta["pair"],
                "start_date": meta["start_date"],
                "end_date": meta["end_date"],
                "timeframe": meta["timeframe"],
                "adr_period": meta["adr_period"],
                "size_mb": round(size_mb, 2),
                "rows": rows,
            })
        return sorted(parquets, key=lambda x: x["name"])

    def list_bridged_parquets(self) -> list[dict]:
        """List existing bridged parquet files with metadata."""
        parquets = []
        for file_path in self.bridged_path.glob("*.parquet"):
            size_mb = file_path.stat().st_size / (1024 * 1024)
            # Get row count
            try:
                df = pl.scan_parquet(file_path)
                rows = df.select(pl.len()).collect().item()
            except Exception:
                rows = 0

            # Parse filename for metadata
            meta = parse_bridged_parquet_filename(file_path.name)

            parquets.append({
                "name": file_path.name,
                "pair": meta["pair"],
                "start_date": meta["start_date"],
                "end_date": meta["end_date"],
                "timeframe": meta["timeframe"],
                "adr_period": meta["adr_period"],
                "size_mb": round(size_mb, 2),
                "rows": rows,
            })
        return sorted(parquets, key=lambda x: (x["pair"] or "", x["timeframe"] or "", x["name"]))

    def _initialize_enriched_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add all 78 enriched columns initialized to null (13 per session x 6 sessions)."""
        new_columns = []
        for prefix in SESSION_PREFIXES.values():
            new_columns.extend([
                # State columns
                pl.lit(None).cast(pl.Int32).alias(f"{prefix}_state_level"),
                pl.lit(None).cast(pl.Utf8).alias(f"{prefix}_state_dir"),
                pl.lit(None).cast(pl.Utf8).alias(f"{prefix}_state_event"),
                # Raw outcome columns
                pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_next"),
                pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_next5"),
                pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_sess"),
                pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_max_up"),
                pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_max_down"),
                # Normalized outcome columns (divided by ADR)
                pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_next_norm"),
                pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_next5_norm"),
                pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_sess_norm"),
                pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_max_up_norm"),
                pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_max_down_norm"),
            ])
        return df.with_columns(new_columns)

    def enrich_file(
        self,
        filename: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> dict:
        """
        Enrich a single raw parquet file.

        Args:
            filename: Name of the parquet file in raw/
            progress_callback: Optional callback(current_day, total_days, message)

        Returns:
            Dict with status, output_filename, rows, days_processed, etc.
        """
        input_path = self.raw_path / filename
        if not input_path.exists():
            return {"status": "error", "message": f"File not found: {filename}"}

        start_time = time_module.time()

        # Load raw data
        df = pl.read_parquet(input_path)
        original_columns = df.columns

        # Validate required columns
        required = ["timestamp", "open", "high", "low", "close"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return {"status": "error", "message": f"Missing required columns: {missing}"}

        # Add date column for grouping
        df = df.with_columns(
            pl.col("timestamp").dt.date().alias("trade_date")
        )

        # Get unique dates sorted
        dates = df.select("trade_date").unique().sort("trade_date")["trade_date"].to_list()
        total_days = len(dates)

        if total_days == 0:
            return {"status": "error", "message": "No data in parquet file"}

        # Initialize enriched columns
        df = self._initialize_enriched_columns(df)

        # Add row index for updating
        df = df.with_row_index("__global_idx")

        # Process each day
        for day_idx, trade_date in enumerate(dates):
            if progress_callback:
                progress_callback(day_idx + 1, total_days, f"Processing {trade_date}")

            # Get day's data
            day_mask = pl.col("trade_date") == trade_date
            day_df = df.filter(day_mask)

            # Get the global indices for this day's rows
            global_indices = day_df["__global_idx"].to_list()

            # Create a minimal day dataframe for processing (without enriched columns)
            day_df_minimal = day_df.select(["timestamp", "open", "high", "low", "close", "volume"] if "volume" in original_columns else ["timestamp", "open", "high", "low", "close"])

            # Process each session
            for session_type in SESSION_PREFIXES.keys():
                processor = SessionProcessor(session_type)
                prefix = SESSION_PREFIXES[session_type]
                adr_col = SESSION_TO_ADR_COL[session_type]

                # Process the day for this session
                session_results = processor.process_day(day_df_minimal, trade_date)

                # Update the main dataframe with results
                for local_idx, (state, outcome) in session_results.items():
                    global_idx = global_indices[local_idx]

                    # Get ADR value for normalization
                    adr_value = day_df[local_idx, adr_col] if adr_col in day_df.columns else None
                    if adr_value is None or adr_value == 0:
                        adr_value = 1.0  # Prevent division by zero

                    # Calculate normalized outcomes
                    out_next_norm = outcome.next_bar_delta / adr_value if outcome.next_bar_delta else 0.0
                    out_next5_norm = outcome.next5_bar_delta / adr_value if outcome.next5_bar_delta else 0.0
                    out_sess_norm = outcome.session_close_delta / adr_value if outcome.session_close_delta else 0.0
                    out_max_up_norm = outcome.session_max_up / adr_value if outcome.session_max_up else 0.0
                    out_max_down_norm = outcome.session_max_down / adr_value if outcome.session_max_down else 0.0

                    # Build update expressions
                    df = df.with_columns([
                        pl.when(pl.col("__global_idx") == global_idx)
                        .then(pl.lit(state.level))
                        .otherwise(pl.col(f"{prefix}_state_level"))
                        .alias(f"{prefix}_state_level"),

                        pl.when(pl.col("__global_idx") == global_idx)
                        .then(pl.lit(state.direction))
                        .otherwise(pl.col(f"{prefix}_state_dir"))
                        .alias(f"{prefix}_state_dir"),

                        pl.when(pl.col("__global_idx") == global_idx)
                        .then(pl.lit(state.event))
                        .otherwise(pl.col(f"{prefix}_state_event"))
                        .alias(f"{prefix}_state_event"),

                        # Raw outcomes
                        pl.when(pl.col("__global_idx") == global_idx)
                        .then(pl.lit(outcome.next_bar_delta))
                        .otherwise(pl.col(f"{prefix}_out_next"))
                        .alias(f"{prefix}_out_next"),

                        pl.when(pl.col("__global_idx") == global_idx)
                        .then(pl.lit(outcome.next5_bar_delta))
                        .otherwise(pl.col(f"{prefix}_out_next5"))
                        .alias(f"{prefix}_out_next5"),

                        pl.when(pl.col("__global_idx") == global_idx)
                        .then(pl.lit(outcome.session_close_delta))
                        .otherwise(pl.col(f"{prefix}_out_sess"))
                        .alias(f"{prefix}_out_sess"),

                        pl.when(pl.col("__global_idx") == global_idx)
                        .then(pl.lit(outcome.session_max_up))
                        .otherwise(pl.col(f"{prefix}_out_max_up"))
                        .alias(f"{prefix}_out_max_up"),

                        pl.when(pl.col("__global_idx") == global_idx)
                        .then(pl.lit(outcome.session_max_down))
                        .otherwise(pl.col(f"{prefix}_out_max_down"))
                        .alias(f"{prefix}_out_max_down"),

                        # Normalized outcomes (divided by ADR)
                        pl.when(pl.col("__global_idx") == global_idx)
                        .then(pl.lit(out_next_norm))
                        .otherwise(pl.col(f"{prefix}_out_next_norm"))
                        .alias(f"{prefix}_out_next_norm"),

                        pl.when(pl.col("__global_idx") == global_idx)
                        .then(pl.lit(out_next5_norm))
                        .otherwise(pl.col(f"{prefix}_out_next5_norm"))
                        .alias(f"{prefix}_out_next5_norm"),

                        pl.when(pl.col("__global_idx") == global_idx)
                        .then(pl.lit(out_sess_norm))
                        .otherwise(pl.col(f"{prefix}_out_sess_norm"))
                        .alias(f"{prefix}_out_sess_norm"),

                        pl.when(pl.col("__global_idx") == global_idx)
                        .then(pl.lit(out_max_up_norm))
                        .otherwise(pl.col(f"{prefix}_out_max_up_norm"))
                        .alias(f"{prefix}_out_max_up_norm"),

                        pl.when(pl.col("__global_idx") == global_idx)
                        .then(pl.lit(out_max_down_norm))
                        .otherwise(pl.col(f"{prefix}_out_max_down_norm"))
                        .alias(f"{prefix}_out_max_down_norm"),
                    ])

        # Drop temporary columns
        df = df.drop(["trade_date", "__global_idx"])

        # Generate output filename
        output_filename = filename.replace(".parquet", "_bridged.parquet")
        output_path = self.bridged_path / output_filename

        # Write output
        df.write_parquet(output_path, compression="zstd")

        elapsed = time_module.time() - start_time

        return {
            "status": "success",
            "message": f"Successfully enriched {filename}",
            "output_filename": output_filename,
            "rows": len(df),
            "days_processed": total_days,
            "processing_time_seconds": round(elapsed, 2),
        }

    def enrich_file_optimized(
        self,
        filename: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> dict:
        """
        Optimized version of enrich_file using batch updates instead of row-by-row.

        Args:
            filename: Name of the parquet file in raw/
            progress_callback: Optional callback(current_day, total_days, message)

        Returns:
            Dict with status, output_filename, rows, days_processed, etc.
        """
        input_path = self.raw_path / filename
        if not input_path.exists():
            return {"status": "error", "message": f"File not found: {filename}"}

        start_time = time_module.time()

        # Load raw data
        df = pl.read_parquet(input_path)
        original_columns = df.columns

        # Validate required columns
        required = ["timestamp", "open", "high", "low", "close"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return {"status": "error", "message": f"Missing required columns: {missing}"}

        # Add row index and date column
        df = df.with_row_index("__global_idx")
        df = df.with_columns(
            pl.col("timestamp").dt.date().alias("trade_date")
        )

        # Get unique dates sorted
        dates = df.select("trade_date").unique().sort("trade_date")["trade_date"].to_list()
        total_days = len(dates)

        if total_days == 0:
            return {"status": "error", "message": "No data in parquet file"}

        # Initialize result storage: dict[global_idx] -> dict[column_name] -> value
        enrichment_data: dict[int, dict[str, any]] = {}

        # Process each day
        for day_idx, trade_date in enumerate(dates):
            if progress_callback:
                progress_callback(day_idx + 1, total_days, f"Processing {trade_date}")

            # Get day's data
            day_df = df.filter(pl.col("trade_date") == trade_date)
            global_indices = day_df["__global_idx"].to_list()

            # Create minimal day dataframe
            cols_to_select = ["timestamp", "open", "high", "low", "close"]
            if "volume" in original_columns:
                cols_to_select.append("volume")
            day_df_minimal = day_df.select(cols_to_select)

            # Process each session
            for session_type in SESSION_PREFIXES.keys():
                processor = SessionProcessor(session_type)
                prefix = SESSION_PREFIXES[session_type]
                adr_col = SESSION_TO_ADR_COL[session_type]
                session_results = processor.process_day(day_df_minimal, trade_date)

                # Store results
                for local_idx, (state, outcome) in session_results.items():
                    global_idx = global_indices[local_idx]

                    if global_idx not in enrichment_data:
                        enrichment_data[global_idx] = {}

                    # Get ADR value for normalization
                    adr_value = day_df[local_idx, adr_col] if adr_col in day_df.columns else None
                    if adr_value is None or adr_value == 0:
                        adr_value = 1.0  # Prevent division by zero

                    # State columns
                    enrichment_data[global_idx][f"{prefix}_state_level"] = state.level
                    enrichment_data[global_idx][f"{prefix}_state_dir"] = state.direction
                    enrichment_data[global_idx][f"{prefix}_state_event"] = state.event

                    # Raw outcome columns
                    enrichment_data[global_idx][f"{prefix}_out_next"] = outcome.next_bar_delta
                    enrichment_data[global_idx][f"{prefix}_out_next5"] = outcome.next5_bar_delta
                    enrichment_data[global_idx][f"{prefix}_out_sess"] = outcome.session_close_delta
                    enrichment_data[global_idx][f"{prefix}_out_max_up"] = outcome.session_max_up
                    enrichment_data[global_idx][f"{prefix}_out_max_down"] = outcome.session_max_down

                    # Normalized outcome columns (divided by ADR)
                    enrichment_data[global_idx][f"{prefix}_out_next_norm"] = outcome.next_bar_delta / adr_value if outcome.next_bar_delta else 0.0
                    enrichment_data[global_idx][f"{prefix}_out_next5_norm"] = outcome.next5_bar_delta / adr_value if outcome.next5_bar_delta else 0.0
                    enrichment_data[global_idx][f"{prefix}_out_sess_norm"] = outcome.session_close_delta / adr_value if outcome.session_close_delta else 0.0
                    enrichment_data[global_idx][f"{prefix}_out_max_up_norm"] = outcome.session_max_up / adr_value if outcome.session_max_up else 0.0
                    enrichment_data[global_idx][f"{prefix}_out_max_down_norm"] = outcome.session_max_down / adr_value if outcome.session_max_down else 0.0

        # Build enrichment columns
        total_rows = len(df)
        column_data = {}

        for prefix in SESSION_PREFIXES.values():
            # State columns
            column_data[f"{prefix}_state_level"] = [None] * total_rows
            column_data[f"{prefix}_state_dir"] = [None] * total_rows
            column_data[f"{prefix}_state_event"] = [None] * total_rows
            # Raw outcome columns
            column_data[f"{prefix}_out_next"] = [None] * total_rows
            column_data[f"{prefix}_out_next5"] = [None] * total_rows
            column_data[f"{prefix}_out_sess"] = [None] * total_rows
            column_data[f"{prefix}_out_max_up"] = [None] * total_rows
            column_data[f"{prefix}_out_max_down"] = [None] * total_rows
            # Normalized outcome columns
            column_data[f"{prefix}_out_next_norm"] = [None] * total_rows
            column_data[f"{prefix}_out_next5_norm"] = [None] * total_rows
            column_data[f"{prefix}_out_sess_norm"] = [None] * total_rows
            column_data[f"{prefix}_out_max_up_norm"] = [None] * total_rows
            column_data[f"{prefix}_out_max_down_norm"] = [None] * total_rows

        # Fill in the enrichment data
        for global_idx, values in enrichment_data.items():
            for col_name, value in values.items():
                column_data[col_name][global_idx] = value

        # Create enrichment DataFrame
        enrichment_df = pl.DataFrame(column_data)

        # Add proper types
        enrichment_df = enrichment_df.with_columns([
            pl.col(c).cast(pl.Int32) for c in enrichment_df.columns if c.endswith("_state_level")
        ])
        enrichment_df = enrichment_df.with_columns([
            pl.col(c).cast(pl.Float64) for c in enrichment_df.columns if c.startswith(tuple(SESSION_PREFIXES.values())) and "_out_" in c
        ])

        # Merge with original
        df = df.drop(["trade_date", "__global_idx"])
        df = pl.concat([df, enrichment_df], how="horizontal")

        # Generate output filename
        output_filename = filename.replace(".parquet", "_bridged.parquet")
        output_path = self.bridged_path / output_filename

        # Write output
        df.write_parquet(output_path, compression="zstd")

        elapsed = time_module.time() - start_time

        return {
            "status": "success",
            "message": f"Successfully enriched {filename}",
            "output_filename": output_filename,
            "rows": len(df),
            "days_processed": total_days,
            "processing_time_seconds": round(elapsed, 2),
        }

    def delete_bridged(self, filename: str) -> dict:
        """Delete a bridged parquet file."""
        file_path = self.bridged_path / filename
        if not file_path.exists():
            return {"status": "error", "message": f"File not found: {filename}"}

        file_path.unlink()
        return {"status": "success", "message": f"Deleted {filename}"}

    def enrich_multiple_files(
        self,
        filenames: list[str],
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> dict:
        """
        Enrich multiple raw parquet files.

        Args:
            filenames: List of parquet filenames in raw/
            progress_callback: Optional callback(current_file, total_files, message)

        Returns:
            Dict with status, results list, and errors list
        """
        results = []
        errors = []
        total = len(filenames)

        for idx, filename in enumerate(filenames):
            if progress_callback:
                progress_callback(idx + 1, total, f"Processing {filename}")

            try:
                result = self.enrich_file_optimized(filename)
                results.append(result)
                if result["status"] == "error":
                    errors.append(f"{filename}: {result['message']}")
            except Exception as e:
                errors.append(f"{filename}: {str(e)}")
                results.append({
                    "status": "error",
                    "message": str(e),
                    "output_filename": None,
                    "rows": 0,
                    "days_processed": 0,
                    "processing_time_seconds": 0.0,
                })

        success_count = sum(1 for r in results if r["status"] == "success")

        return {
            "status": "success" if success_count == total else "partial" if success_count > 0 else "error",
            "message": f"Enriched {success_count}/{total} files",
            "results": results,
            "errors": errors,
        }
