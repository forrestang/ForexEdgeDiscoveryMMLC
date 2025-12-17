import polars as pl
from datetime import timedelta


def est_to_utc(df: pl.DataFrame) -> pl.DataFrame:
    """
    Convert EST timestamps to UTC.
    EST is always UTC-5 (no DST consideration per requirements).
    This means we add 5 hours to convert EST -> UTC.
    """
    return df.with_columns(
        (pl.col("timestamp") + timedelta(hours=5)).alias("timestamp")
    )
