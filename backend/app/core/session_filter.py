import polars as pl
from datetime import date, time, datetime

from app.config import SESSION_TIMES, SessionType


def filter_by_session(df: pl.DataFrame, session: SessionType) -> pl.DataFrame:
    """
    Filter DataFrame to include only bars within session time boundaries.

    Args:
        df: DataFrame with timestamp column
        session: Session type (full_day, asia, london, ny)

    Returns:
        Filtered DataFrame
    """
    (start_hour, start_min), (end_hour, end_min) = SESSION_TIMES[session]
    start_time = time(start_hour, start_min)
    end_time = time(end_hour, end_min)

    return df.filter(
        (pl.col("timestamp").dt.time() >= start_time)
        & (pl.col("timestamp").dt.time() < end_time)
    )


def filter_by_date(df: pl.DataFrame, target_date: date) -> pl.DataFrame:
    """
    Filter DataFrame to include only bars from a specific date.

    Args:
        df: DataFrame with timestamp column
        target_date: The date to filter for

    Returns:
        Filtered DataFrame
    """
    return df.filter(pl.col("timestamp").dt.date() == target_date)


def filter_by_date_and_session(
    df: pl.DataFrame, target_date: date, session: SessionType
) -> pl.DataFrame:
    """
    Filter DataFrame by both date and session.

    Args:
        df: DataFrame with timestamp column
        target_date: The date to filter for
        session: Session type (full_day, asia, london, ny)

    Returns:
        Filtered DataFrame
    """
    df = filter_by_date(df, target_date)
    df = filter_by_session(df, session)
    return df


def get_available_dates(df: pl.DataFrame) -> list[date]:
    """Get list of unique dates available in the DataFrame."""
    return df.select(pl.col("timestamp").dt.date().unique()).to_series().to_list()
