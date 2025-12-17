"""
Future Truth computation for Edge Finder ground truth labels.

Computes "what happened next" for each bar snapshot, providing
the ground truth labels for model training and edge probability aggregation.

Metrics computed:
- Immediate Edge: Next bar's move (ATR normalized)
- Session Drift: End-of-session price relative to current (ATR normalized)
- MAE: Maximum Adverse Excursion before session end
- MFE: Maximum Favorable Excursion before session end
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional
import numpy as np
import polars as pl


@dataclass
class BarMetadata:
    """
    Ground truth labels computed for each bar snapshot.

    All price movements are normalized by session-start ATR,
    enabling comparison across pairs and volatility regimes.
    """
    session_id: str           # e.g., "EURUSD_2024-01-15_london_M5"
    bar_index: int            # 0-indexed position in session

    # Immediate edge (next bar)
    next_bar_move_atr: float  # (next_close - current_close) / ATR

    # Session-level outcomes
    session_drift_atr: float  # (session_end_close - current_close) / ATR
    mae_to_session_end: float # Maximum Adverse Excursion (negative = against long)
    mfe_to_session_end: float # Maximum Favorable Excursion (positive = profit if long)

    # Position context
    bars_remaining: int       # How many bars until session end
    session_progress: float   # bar_index / total_bars (0-1)


@dataclass
class SessionDataset:
    """
    Complete training sample for one session.

    Contains the waveform matrix and all ground truth metadata
    needed for VAE training and edge probability computation.
    """
    session_id: str
    pair: str
    session_date: date
    session_type: str         # "asia", "london", "ny", "full_day"
    timeframe: str            # "M5", "M10", etc.
    matrix: np.ndarray        # Shape: [num_bars, 20]
    metadata: list[BarMetadata]
    session_start_atr: float
    total_bars: int


def generate_session_id(
    pair: str,
    session_date: date,
    session_type: str,
    timeframe: str
) -> str:
    """
    Generate a unique session identifier.

    Format: {PAIR}_{DATE}_{SESSION}_{TIMEFRAME}
    Example: EURUSD_2024-01-15_london_M5
    """
    date_str = session_date.isoformat()
    return f"{pair}_{date_str}_{session_type}_{timeframe}"


def compute_bar_metadata(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    session_start_atr: float,
    session_id: str,
) -> list[BarMetadata]:
    """
    Compute ground truth metadata for every bar in a session.

    Args:
        closes: Array of close prices for the session
        highs: Array of high prices
        lows: Array of low prices
        session_start_atr: ATR for normalization
        session_id: Unique session identifier

    Returns:
        List of BarMetadata, one per bar
    """
    n_bars = len(closes)
    if n_bars == 0:
        return []

    metadata = []
    session_end_close = closes[-1]

    for i in range(n_bars):
        current_close = closes[i]
        bars_remaining = n_bars - i - 1

        # Next bar move (0 for last bar)
        if i < n_bars - 1:
            next_bar_move = (closes[i + 1] - current_close) / session_start_atr
        else:
            next_bar_move = 0.0

        # Session drift
        session_drift = (session_end_close - current_close) / session_start_atr

        # MAE/MFE from current bar to session end
        if bars_remaining > 0:
            future_closes = closes[i + 1:]
            future_highs = highs[i + 1:]
            future_lows = lows[i + 1:]

            # MAE: worst drawdown (most negative if long, most positive if short)
            # For a hypothetical LONG position at current_close:
            # MAE = (lowest_future_low - current_close) / ATR
            mae = (np.min(future_lows) - current_close) / session_start_atr

            # MFE: best excursion
            # MFE = (highest_future_high - current_close) / ATR
            mfe = (np.max(future_highs) - current_close) / session_start_atr
        else:
            mae = 0.0
            mfe = 0.0

        metadata.append(BarMetadata(
            session_id=session_id,
            bar_index=i,
            next_bar_move_atr=next_bar_move,
            session_drift_atr=session_drift,
            mae_to_session_end=mae,
            mfe_to_session_end=mfe,
            bars_remaining=bars_remaining,
            session_progress=i / max(n_bars - 1, 1),
        ))

    return metadata


def compute_session_dataset(
    df: pl.DataFrame,
    matrix: np.ndarray,
    session_start_atr: float,
    pair: str,
    session_date: date,
    session_type: str,
    timeframe: str,
) -> SessionDataset:
    """
    Build a complete SessionDataset from raw data and computed matrix.

    Args:
        df: DataFrame with [timestamp, open, high, low, close]
        matrix: Pre-computed waveform matrix [num_bars, 20]
        session_start_atr: ATR for the session
        pair: Currency pair (e.g., "EURUSD")
        session_date: Date of the session
        session_type: Session type (e.g., "london")
        timeframe: Timeframe (e.g., "M10")

    Returns:
        SessionDataset: Complete training sample
    """
    session_id = generate_session_id(pair, session_date, session_type, timeframe)

    closes = df["close"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    metadata = compute_bar_metadata(
        closes=closes,
        highs=highs,
        lows=lows,
        session_start_atr=session_start_atr,
        session_id=session_id,
    )

    return SessionDataset(
        session_id=session_id,
        pair=pair,
        session_date=session_date,
        session_type=session_type,
        timeframe=timeframe,
        matrix=matrix,
        metadata=metadata,
        session_start_atr=session_start_atr,
        total_bars=len(closes),
    )
