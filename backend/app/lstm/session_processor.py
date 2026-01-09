"""
Session Processor - Per-session MMLC processing for LSTM Bridge

This module handles running the MMLC core on individual trading sessions,
creating a fresh MMLC instance for each session on each day.
"""

from datetime import date, time, datetime
from typing import Optional
import polars as pl

from app.config import SESSION_TIMES, SessionType
from app.lstm.mmlc_core import MMLCCore, Candle, BarState, BarOutcome


# Map session type to column prefix
SESSION_PREFIXES = {
    "asia": "asia",
    "london": "lon",
    "ny": "ny",
    "full_day": "day",
    # Combo sessions
    "asia_london": "asialon",
    "london_ny": "lonny",
}


class SessionProcessor:
    """
    Processes a single session (Asia, London, NY, or Full Day) for one day.
    Creates a fresh MMLCCore instance for each session.
    """

    def __init__(self, session_type: SessionType):
        """
        Initialize the processor for a specific session type.

        Args:
            session_type: One of "asia", "london", "ny", "full_day"
        """
        self.session_type = session_type
        self.prefix = SESSION_PREFIXES[session_type]
        self._start_time, self._end_time = self._get_session_times()

    def _get_session_times(self) -> tuple[time, time]:
        """Get session start and end times."""
        (start_h, start_m), (end_h, end_m) = SESSION_TIMES[self.session_type]
        return time(start_h, start_m), time(end_h, end_m)

    def is_bar_in_session(self, bar_time: time) -> bool:
        """Check if a bar's time falls within this session."""
        return self._start_time <= bar_time < self._end_time

    def filter_session_bars(self, day_df: pl.DataFrame) -> tuple[pl.DataFrame, list[int]]:
        """
        Filter a day's DataFrame to only include bars within this session.

        Args:
            day_df: DataFrame with all bars for one day

        Returns:
            Tuple of (filtered DataFrame, list of original row indices)
        """
        # Add row index for mapping back
        day_df = day_df.with_row_index("__original_idx")

        # Extract time from timestamp
        session_df = day_df.filter(
            (pl.col("timestamp").dt.time() >= self._start_time) &
            (pl.col("timestamp").dt.time() < self._end_time)
        )

        # Get the original indices
        original_indices = session_df["__original_idx"].to_list()

        # Remove the temporary column
        session_df = session_df.drop("__original_idx")

        return session_df, original_indices

    def df_to_candles(self, df: pl.DataFrame) -> list[Candle]:
        """Convert a Polars DataFrame to a list of Candle objects."""
        candles = []
        for row in df.iter_rows(named=True):
            candles.append(Candle(
                timestamp=row["timestamp"],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row.get("volume", 0)
            ))
        return candles

    def process_day(
        self,
        day_df: pl.DataFrame,
        trade_date: date
    ) -> dict[int, tuple[BarState, BarOutcome]]:
        """
        Process all bars for a single day's session.

        Args:
            day_df: DataFrame with all bars for this day
            trade_date: The date being processed

        Returns:
            Dict mapping original day_df row index to (BarState, BarOutcome) tuple
        """
        # Filter to session time window
        session_df, original_indices = self.filter_session_bars(day_df)

        if session_df.is_empty():
            return {}

        # Convert to Candle objects
        candles = self.df_to_candles(session_df)

        # Create fresh MMLC instance and process
        mmlc = MMLCCore()
        bar_states = mmlc.process_session(candles)
        bar_outcomes = mmlc.calculate_outcomes(candles)

        # Map session bar index to original day_df index
        result = {}
        for session_idx, (state, outcome) in enumerate(zip(bar_states, bar_outcomes)):
            original_idx = original_indices[session_idx]
            result[original_idx] = (state, outcome)

        return result


def process_day_all_sessions(
    day_df: pl.DataFrame,
    trade_date: date
) -> dict[str, dict[int, tuple[BarState, BarOutcome]]]:
    """
    Process all 6 sessions for a single day.

    Args:
        day_df: DataFrame with all bars for this day
        trade_date: The date being processed

    Returns:
        Dict mapping session prefix to {row_idx: (BarState, BarOutcome)}
    """
    results = {}

    for session_type in SESSION_PREFIXES.keys():
        processor = SessionProcessor(session_type)
        session_results = processor.process_day(day_df, trade_date)
        results[SESSION_PREFIXES[session_type]] = session_results

    return results
