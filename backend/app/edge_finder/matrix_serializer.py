"""
Matrix Serializer for converting waveform snapshots to fixed-size tensors.

Converts the hierarchical wave stack at each bar into a [num_bars, 20] matrix
suitable for VAE training and inference.

Matrix Schema (20 channels = 5 levels x 4 features):
- Channels 0-3:   L1 [Direction, Amplitude, Duration, Leg_Count]
- Channels 4-7:   L2 [Direction, Amplitude, Duration, Leg_Count]
- Channels 8-11:  L3 [Direction, Amplitude, Duration, Leg_Count]
- Channels 12-15: L4 [Direction, Amplitude, Duration, Leg_Count]
- Channels 16-19: L5 [Direction, Amplitude, Duration, Leg_Count]

Normalization:
- Direction: +1 (UP), -1 (DOWN), 0 (inactive)
- Amplitude: Normalized by session-start ATR (shape-based learning)
- Duration: bars_since_start / total_session_bars (0-1 range)
- Leg_Count: count / expected_max (0-1 range, expected_max varies by level)
"""

import numpy as np
import polars as pl
from typing import Optional

from app.waveform.state_snapshot import StackSnapshot, WaveSnapshot


# Expected maximum leg counts per level (for normalization)
# These are empirical values - L1 tends to have fewer swings, L5 many more
EXPECTED_MAX_LEG_COUNTS = {
    1: 10,   # L1: Backbone waves, relatively few per session
    2: 20,   # L2: More frequent
    3: 40,   # L3: Higher frequency
    4: 80,   # L4: Even higher
    5: 160,  # L5: Most granular
}


def compute_session_start_atr(
    df: pl.DataFrame,
    lookback: int = 14
) -> float:
    """
    Compute ATR from the first N bars of a session.

    This ATR is used to normalize all amplitude values, enabling
    the model to learn shapes rather than absolute price movements.

    Args:
        df: DataFrame with columns [timestamp, open, high, low, close]
        lookback: Number of bars to use for ATR calculation

    Returns:
        float: Average True Range for session normalization
    """
    if len(df) == 0:
        return 1.0  # Fallback to prevent division by zero

    # Use available bars if fewer than lookback
    n_bars = min(lookback, len(df))
    subset = df.head(n_bars)

    # True Range = max(H-L, |H-prev_close|, |L-prev_close|)
    # For first bar, we just use H-L
    highs = subset["high"].to_numpy()
    lows = subset["low"].to_numpy()
    closes = subset["close"].to_numpy()

    tr_values = []

    for i in range(n_bars):
        hl_range = highs[i] - lows[i]

        if i == 0:
            tr = hl_range
        else:
            prev_close = closes[i - 1]
            hc_range = abs(highs[i] - prev_close)
            lc_range = abs(lows[i] - prev_close)
            tr = max(hl_range, hc_range, lc_range)

        tr_values.append(tr)

    atr = np.mean(tr_values) if tr_values else 1.0

    # Ensure non-zero ATR
    return max(atr, 1e-8)


def snapshot_to_row(
    snapshot: StackSnapshot,
    session_start_atr: float,
    total_session_bars: int,
) -> np.ndarray:
    """
    Convert a single StackSnapshot to a 20-element feature row.

    Args:
        snapshot: The wave stack state at a specific bar
        session_start_atr: ATR for amplitude normalization
        total_session_bars: Total bars in session for duration normalization

    Returns:
        np.ndarray: Shape (20,) with features for L1-L5
    """
    row = np.zeros(20, dtype=np.float32)

    for wave in snapshot.waves:
        if wave.level < 1 or wave.level > 5:
            continue  # Skip unexpected levels

        # Calculate channel offset for this level
        idx = (wave.level - 1) * 4

        # Feature 0: Direction (+1, -1)
        row[idx] = float(wave.direction)

        # Feature 1: Amplitude (ATR-normalized)
        row[idx + 1] = wave.amplitude / session_start_atr

        # Feature 2: Duration (normalized 0-1)
        duration_norm = wave.duration_bars / max(total_session_bars, 1)
        row[idx + 2] = min(duration_norm, 1.0)  # Cap at 1.0

        # Feature 3: Leg Count (normalized 0-1)
        leg_count = snapshot.get_leg_count(wave.level)
        expected_max = EXPECTED_MAX_LEG_COUNTS.get(wave.level, 100)
        row[idx + 3] = min(leg_count / expected_max, 1.0)  # Cap at 1.0

    return row


def snapshots_to_matrix(
    snapshots: list[StackSnapshot],
    session_start_atr: float,
) -> np.ndarray:
    """
    Convert a full session's snapshots to a [num_bars, 20] matrix.

    Args:
        snapshots: List of StackSnapshot, one per bar
        session_start_atr: ATR for amplitude normalization

    Returns:
        np.ndarray: Shape (num_bars, 20) with all bar features
    """
    if not snapshots:
        return np.zeros((0, 20), dtype=np.float32)

    total_bars = len(snapshots)
    matrix = np.zeros((total_bars, 20), dtype=np.float32)

    for i, snapshot in enumerate(snapshots):
        matrix[i] = snapshot_to_row(snapshot, session_start_atr, total_bars)

    return matrix


def get_session_bar_count(session: str, timeframe: str) -> int:
    """
    Calculate expected bar count for a session/timeframe combination.

    Args:
        session: One of "asia", "london", "ny", "full_day"
        timeframe: One of "M1", "M5", "M10", "M15", "M30", "H1", "H4"

    Returns:
        int: Expected number of bars
    """
    # Session durations in hours
    session_hours = {
        "asia": 9,      # 00:00 - 09:00 UTC
        "london": 9,    # 08:00 - 17:00 UTC
        "ny": 9,        # 13:00 - 22:00 UTC
        "full_day": 22, # 00:00 - 22:00 UTC
    }

    # Timeframe in minutes
    timeframe_minutes = {
        "M1": 1,
        "M5": 5,
        "M10": 10,
        "M15": 15,
        "M30": 30,
        "H1": 60,
        "H4": 240,
    }

    hours = session_hours.get(session, 22)
    minutes_per_bar = timeframe_minutes.get(timeframe, 10)

    total_minutes = hours * 60
    return total_minutes // minutes_per_bar
