import polars as pl
from pathlib import Path
from datetime import date
from typing import Optional

from app.config import settings, TIMEFRAMES


def ensure_cache_directory(cache_path: Optional[Path] = None) -> Path:
    """Create cache directory if it doesn't exist."""
    path = cache_path or settings.cache_path
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_parquet_path(cache_path: Path, pair: str, timeframe: str) -> Path:
    """Get the parquet file path for a pair/timeframe combination."""
    return cache_path / f"{pair}_{timeframe}.parquet"


def save_to_cache(
    df: pl.DataFrame, pair: str, timeframe: str, cache_path: Optional[Path] = None
) -> Path:
    """Save DataFrame to parquet cache."""
    path = cache_path or settings.cache_path
    ensure_cache_directory(path)

    file_path = get_parquet_path(path, pair, timeframe)
    df.write_parquet(file_path, compression="zstd")
    return file_path


def load_from_cache(
    pair: str, timeframe: str, cache_path: Optional[Path] = None
) -> Optional[pl.DataFrame]:
    """Load DataFrame from parquet cache."""
    path = cache_path or settings.cache_path
    file_path = get_parquet_path(path, pair, timeframe)

    if not file_path.exists():
        return None

    return pl.read_parquet(file_path)


def get_cached_instruments(cache_path: Optional[Path] = None) -> list[dict]:
    """
    Get list of all cached instruments with their metadata.

    Returns list of dicts with:
        - pair: str
        - timeframes: list[str]
        - start_date: date
        - end_date: date
        - file_count: int
    """
    path = cache_path or settings.cache_path

    if not path.exists():
        return []

    # Group parquet files by pair
    pairs_data: dict[str, dict] = {}

    for parquet_file in path.glob("*.parquet"):
        # Parse filename: {PAIR}_{TIMEFRAME}.parquet
        parts = parquet_file.stem.rsplit("_", 1)
        if len(parts) != 2:
            continue

        pair, timeframe = parts
        if timeframe not in TIMEFRAMES:
            continue

        if pair not in pairs_data:
            pairs_data[pair] = {
                "pair": pair,
                "timeframes": [],
                "start_date": None,
                "end_date": None,
                "file_count": 0,
            }

        pairs_data[pair]["timeframes"].append(timeframe)
        pairs_data[pair]["file_count"] += 1

        # Load M1 data to get date range (only once per pair)
        if timeframe == "M1" and pairs_data[pair]["start_date"] is None:
            df = pl.read_parquet(parquet_file)
            if len(df) > 0:
                pairs_data[pair]["start_date"] = df["timestamp"].min().date()
                pairs_data[pair]["end_date"] = df["timestamp"].max().date()

    # Sort timeframes for each pair
    for pair in pairs_data:
        pairs_data[pair]["timeframes"].sort(
            key=lambda x: TIMEFRAMES.index(x) if x in TIMEFRAMES else 999
        )

    return list(pairs_data.values())


def clear_pair_cache(pair: str, cache_path: Optional[Path] = None) -> int:
    """Remove all cached files for a pair. Returns number of files removed."""
    path = cache_path or settings.cache_path
    count = 0

    for parquet_file in path.glob(f"{pair}_*.parquet"):
        parquet_file.unlink()
        count += 1

    return count
