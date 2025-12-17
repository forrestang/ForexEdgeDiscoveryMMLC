import polars as pl
from app.config import TIMEFRAME_INTERVALS, TimeframeType


def upsample_ohlc(df: pl.DataFrame, target_timeframe: TimeframeType) -> pl.DataFrame:
    """
    Resample M1 data to higher timeframes.

    Args:
        df: DataFrame with M1 OHLC data (columns: timestamp, open, high, low, close, volume)
        target_timeframe: Target timeframe (M5, M10, M15, M30, H1, H4)

    Returns:
        Resampled DataFrame
    """
    if target_timeframe == "M1":
        return df

    interval = TIMEFRAME_INTERVALS[target_timeframe]

    return df.group_by_dynamic(
        "timestamp",
        every=interval,
        closed="left",
        label="left",
    ).agg(
        [
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
        ]
    )


def generate_all_timeframes(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """
    Generate all timeframe aggregations from M1 data.

    Args:
        df: DataFrame with M1 OHLC data

    Returns:
        Dict mapping timeframe name to resampled DataFrame
    """
    timeframes = ["M1", "M5", "M10", "M15", "M30", "H1", "H4"]
    result = {}

    for tf in timeframes:
        result[tf] = upsample_ohlc(df, tf)

    return result
