"""
Session Dataset Generator for Edge Finder.

Processes historical OHLC data and generates session datasets
with waveform matrices and future truth metadata.

This is the main ETL pipeline for the Edge Finder system.
"""

from datetime import date
from pathlib import Path
from typing import Optional, Generator
import polars as pl

from app.config import settings, SessionType, TimeframeType
from app.core.cache_manager import load_from_cache, get_cached_instruments
from app.core.session_filter import filter_by_date_and_session, get_available_dates
from app.waveform.streaming_engine import StreamingWaveformEngine
from app.edge_finder.matrix_serializer import (
    snapshots_to_matrix,
    compute_session_start_atr,
)
from app.edge_finder.future_truth import compute_session_dataset, SessionDataset
from app.edge_finder.storage import save_session_dataset, list_session_files


def generate_session_dataset(
    df: pl.DataFrame,
    pair: str,
    session_date: date,
    session_type: SessionType,
    timeframe: TimeframeType,
) -> Optional[SessionDataset]:
    """
    Generate a single SessionDataset from filtered OHLC data.

    Args:
        df: DataFrame with [timestamp, open, high, low, close] for one session
        pair: Currency pair (e.g., "EURUSD")
        session_date: Date of the session
        session_type: Session type (e.g., "ny")
        timeframe: Timeframe (e.g., "M10")

    Returns:
        SessionDataset or None if data is insufficient
    """
    if len(df) < 5:  # Minimum bars for meaningful waveform
        return None

    # Sort by timestamp to ensure correct order
    df = df.sort("timestamp")

    # Compute session-start ATR
    session_start_atr = compute_session_start_atr(df, lookback=14)

    # Run streaming waveform engine to get snapshots
    engine = StreamingWaveformEngine()
    waves, snapshots = engine.process_session_with_snapshots(df)

    if not snapshots:
        return None

    # Convert snapshots to matrix
    matrix = snapshots_to_matrix(snapshots, session_start_atr)

    # Build complete dataset with future truth
    dataset = compute_session_dataset(
        df=df,
        matrix=matrix,
        session_start_atr=session_start_atr,
        pair=pair,
        session_date=session_date,
        session_type=session_type,
        timeframe=timeframe,
    )

    return dataset


def iter_sessions(
    pair: str,
    timeframe: TimeframeType,
    session_type: SessionType,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    working_directory: Optional[Path] = None,
) -> Generator[tuple[date, pl.DataFrame], None, None]:
    """
    Iterate over all available sessions for a pair/timeframe/session combination.

    Args:
        pair: Currency pair
        timeframe: Timeframe to use
        session_type: Session type
        start_date: Optional start date filter
        end_date: Optional end date filter
        working_directory: Optional override for cache directory

    Yields:
        Tuples of (session_date, session_df)
    """
    cache_path = (working_directory or settings.default_working_directory) / "cache"

    # Load cached data
    df = load_from_cache(pair, timeframe, cache_path)
    if df is None:
        return

    # Get available dates
    available_dates = get_available_dates(df)

    for session_date in sorted(available_dates):
        # Apply date filters
        if start_date and session_date < start_date:
            continue
        if end_date and session_date > end_date:
            continue

        # Filter to this date and session
        session_df = filter_by_date_and_session(df, session_date, session_type)

        if len(session_df) > 0:
            yield session_date, session_df


def generate_all_sessions(
    pair: str,
    timeframe: TimeframeType,
    session_type: SessionType,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    working_directory: Optional[Path] = None,
    skip_existing: bool = True,
) -> dict:
    """
    Generate and save session datasets for a pair/timeframe/session combination.

    Args:
        pair: Currency pair
        timeframe: Timeframe
        session_type: Session type
        start_date: Optional start date filter
        end_date: Optional end date filter
        working_directory: Optional override for directories
        skip_existing: If True, skip sessions that already have saved datasets

    Returns:
        Dict with generation statistics
    """
    stats = {
        "processed": 0,
        "saved": 0,
        "skipped": 0,
        "errors": 0,
        "error_details": [],
    }

    # Get existing sessions to skip
    existing = set()
    if skip_existing:
        existing = set(list_session_files(working_directory, pair, session_type, timeframe))

    for session_date, session_df in iter_sessions(
        pair=pair,
        timeframe=timeframe,
        session_type=session_type,
        start_date=start_date,
        end_date=end_date,
        working_directory=working_directory,
    ):
        stats["processed"] += 1

        # Build expected session_id to check if exists
        expected_id = f"{pair}_{session_date.isoformat()}_{session_type}_{timeframe}"
        if expected_id in existing:
            stats["skipped"] += 1
            continue

        try:
            dataset = generate_session_dataset(
                df=session_df,
                pair=pair,
                session_date=session_date,
                session_type=session_type,
                timeframe=timeframe,
            )

            if dataset is not None:
                save_session_dataset(dataset, working_directory)
                stats["saved"] += 1
            else:
                stats["skipped"] += 1

        except Exception as e:
            stats["errors"] += 1
            stats["error_details"].append({
                "date": session_date.isoformat(),
                "error": str(e),
            })

    return stats


def generate_test_dataset(
    pair: str = "EURUSD",
    timeframe: TimeframeType = "M10",
    session_type: SessionType = "ny",
    max_sessions: int = 100,
    working_directory: Optional[Path] = None,
) -> dict:
    """
    Generate a test dataset with a limited number of sessions.

    This is used for Phase 0 validation before full dataset generation.

    Args:
        pair: Currency pair (default: EURUSD)
        timeframe: Timeframe (default: M10)
        session_type: Session type (default: ny)
        max_sessions: Maximum sessions to generate (default: 100)
        working_directory: Optional override for directories

    Returns:
        Dict with generation statistics
    """
    stats = {
        "processed": 0,
        "saved": 0,
        "skipped": 0,
        "errors": 0,
        "error_details": [],
    }

    for session_date, session_df in iter_sessions(
        pair=pair,
        timeframe=timeframe,
        session_type=session_type,
        working_directory=working_directory,
    ):
        if stats["saved"] >= max_sessions:
            break

        stats["processed"] += 1

        try:
            dataset = generate_session_dataset(
                df=session_df,
                pair=pair,
                session_date=session_date,
                session_type=session_type,
                timeframe=timeframe,
            )

            if dataset is not None:
                save_session_dataset(dataset, working_directory)
                stats["saved"] += 1
            else:
                stats["skipped"] += 1

        except Exception as e:
            stats["errors"] += 1
            stats["error_details"].append({
                "date": session_date.isoformat(),
                "error": str(e),
            })

    return stats


def get_available_pairs(working_directory: Optional[Path] = None) -> list[str]:
    """Get list of pairs with cached data."""
    cache_path = (working_directory or settings.default_working_directory) / "cache"
    instruments = get_cached_instruments(cache_path)
    return [inst["pair"] for inst in instruments]
