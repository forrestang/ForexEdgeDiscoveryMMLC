import polars as pl
from pathlib import Path
import re
from typing import Optional


def extract_pair_from_filename(filename: str) -> Optional[str]:
    """Extract currency pair from DAT_MT_[PAIR]_M1_[YEAR].csv or DAT_NT_[PAIR]_M1_[YEAR].csv format."""
    match = re.match(r"DAT_[MN]T_(\w+)_M1_\d+\.csv", filename, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def parse_metatrader_csv(file_path: Path) -> pl.DataFrame:
    """
    Parse MetaTrader/NinjaTrader CSV file.
    Supports format: YYYYMMDD HHMMSS;Open;High;Low;Close;Volume (semicolon-separated, no header)
    """
    df = pl.read_csv(
        file_path,
        has_header=False,
        separator=";",
        new_columns=["datetime_str", "open", "high", "low", "close", "volume"],
        dtypes={
            "datetime_str": pl.Utf8,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Int64,
        },
    )

    # Parse datetime from YYYYMMDD HHMMSS format
    df = df.with_columns(
        pl.col("datetime_str")
        .str.strptime(pl.Datetime("ms"), "%Y%m%d %H%M%S")
        .alias("timestamp")
    )

    return df.select(["timestamp", "open", "high", "low", "close", "volume"])


def scan_source_folder(data_path: Path) -> dict[str, list[Path]]:
    """
    Scan the data folder for MetaTrader CSV files.
    Returns dict mapping pair names to list of file paths.
    """
    if not data_path.exists():
        return {}

    files_by_pair: dict[str, list[Path]] = {}

    # Match both DAT_MT_ and DAT_NT_ prefixes
    for file_path in list(data_path.glob("DAT_MT_*_M1_*.csv")) + list(data_path.glob("DAT_NT_*_M1_*.csv")):
        pair = extract_pair_from_filename(file_path.name)
        if pair:
            if pair not in files_by_pair:
                files_by_pair[pair] = []
            files_by_pair[pair].append(file_path)

    # Sort files by name (chronological order based on year in filename)
    for pair in files_by_pair:
        files_by_pair[pair].sort(key=lambda p: p.name)

    return files_by_pair


def load_and_stitch_pair_data(file_paths: list[Path]) -> pl.DataFrame:
    """
    Load multiple CSV files for a pair and stitch them together.
    Deduplicates overlapping records.
    """
    if not file_paths:
        return pl.DataFrame()

    dfs = []
    for fp in file_paths:
        df = parse_metatrader_csv(fp)
        dfs.append(df)

    # Concatenate all dataframes
    combined = pl.concat(dfs)

    # Sort by timestamp and deduplicate
    combined = combined.sort("timestamp").unique(subset=["timestamp"], keep="first")

    return combined
