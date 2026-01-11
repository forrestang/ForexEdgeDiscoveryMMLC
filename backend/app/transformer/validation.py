"""
Synthetic data generation and validation for Transformer model.

Creates test parquets with known input→output patterns to verify:
1. Sanity Check: Model can map single input to output
2. Memory Test: Model attention can look back to sequence start
3. Logic Test: Model can combine multiple features

Two generation modes:
- Pure synthetic: Generates completely artificial OHLC data
- From source: Uses real parquet data as base, only replaces state/outcome columns

Usage:
    python -m app.transformer.validation generate --output-dir /path/to/working_dir
    python -m app.transformer.validation run --working-dir /path/to/working_dir --test sanity
"""

import argparse
import sys
import polars as pl
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

# Force unbuffered output for better logging
sys.stdout.reconfigure(line_buffering=True)

# Session prefixes and their ADR column names
SESSION_PREFIXES = ["asia", "lon", "ny", "day", "asialon", "lonny"]
SESSION_ADR_COLS = {
    "asia": "adr_asia",
    "lon": "adr_london",
    "ny": "adr_ny",
    "day": "adr_full_day",
    "asialon": "adr_asia_london",
    "lonny": "adr_london_ny",
}

# Session durations in hours (for calculating bars per session)
SESSION_HOURS = {
    "asia": 9,
    "lon": 9,
    "ny": 9,
    "day": 22,
    "asialon": 17,  # 00:00 to 17:00 = 17 hours
    "lonny": 14,    # 08:00 to 22:00 = 14 hours
}

# Session time windows (UTC) - start hour, end hour
SESSION_TIME_WINDOWS = {
    "asia": (0, 9),       # 00:00 - 09:00 UTC
    "lon": (8, 17),       # 08:00 - 17:00 UTC
    "ny": (13, 22),       # 13:00 - 22:00 UTC
    "day": (0, 22),       # 00:00 - 22:00 UTC
    "asialon": (0, 17),   # 00:00 - 17:00 UTC
    "lonny": (8, 22),     # 08:00 - 22:00 UTC
}

# Timeframe in minutes
TIMEFRAME_MINUTES = {
    "M1": 1,
    "M5": 5,
    "M10": 10,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
}

# Default session counts per test type
DEFAULT_SESSIONS = {
    "sanity": 1000,
    "memory": 2000,
    "logic": 5000,
}


def get_bars_per_session(timeframe: str, session_hours: int = 9) -> int:
    """Calculate number of bars in a session for given timeframe."""
    minutes_per_bar = TIMEFRAME_MINUTES.get(timeframe, 5)
    total_minutes = session_hours * 60
    return total_minutes // minutes_per_bar


def create_base_schema(
    n_rows: int,
    adr_value: float = 1.0,
    timeframe: str = "M5",
    seed: int = 42,
) -> pl.DataFrame:
    """
    Create a DataFrame with the full enriched parquet schema.

    Args:
        n_rows: Number of rows to create
        adr_value: ADR value (e.g., 0.0050 for 50 pips)
        timeframe: Timeframe for timestamp intervals
        seed: Random seed for reproducibility

    Returns:
        Polars DataFrame with all required columns initialized
    """
    np.random.seed(seed)

    # Get minutes per bar from timeframe
    minutes_per_bar = TIMEFRAME_MINUTES.get(timeframe, 5)

    # Timestamp column based on timeframe
    start_date = datetime(2022, 1, 1)
    timestamps = [start_date + timedelta(minutes=minutes_per_bar * i) for i in range(n_rows)]

    # Base OHLC (random noise - not used for pattern but needed for schema)
    base_price = 1.1000
    opens = base_price + np.random.normal(0, 0.001, n_rows)
    closes = opens + np.random.normal(0, 0.0005, n_rows)
    highs = np.maximum(opens, closes) + 0.0001
    lows = np.minimum(opens, closes) - 0.0001

    # Start building DataFrame
    df = pl.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
    })

    # Add ADR columns with user-specified value
    for prefix, adr_col in SESSION_ADR_COLS.items():
        df = df.with_columns(pl.lit(adr_value).alias(adr_col))

    # Add state/outcome columns for all sessions (initialized to defaults)
    for prefix in SESSION_PREFIXES:
        df = df.with_columns([
            # State columns
            pl.lit(1).cast(pl.Int32).alias(f"{prefix}_state_level"),
            pl.lit("EXTENSION").alias(f"{prefix}_state_event"),
            pl.lit("UP").alias(f"{prefix}_state_dir"),
            # Outcome columns (raw)
            pl.lit(0.0).alias(f"{prefix}_out_next"),
            pl.lit(0.0).alias(f"{prefix}_out_sess"),
            pl.lit(0.0).alias(f"{prefix}_out_max_up"),
            pl.lit(0.0).alias(f"{prefix}_out_max_down"),
            # Outcome columns (normalized by ADR)
            pl.lit(0.0).alias(f"{prefix}_out_next_norm"),
            pl.lit(0.0).alias(f"{prefix}_out_sess_norm"),
            pl.lit(0.0).alias(f"{prefix}_out_max_up_norm"),
            pl.lit(0.0).alias(f"{prefix}_out_max_down_norm"),
        ])

    return df


def generate_sanity_check(
    output_dir: Path,
    n_sessions: int = 1000,
    timeframe: str = "M5",
    adr_value: float = 1.0,
    target_outcome: str = "max_up",
    seed: int = 42,
) -> Path:
    """
    Generate Test 1: Sanity Check (The Repeater).

    Rule:
        state_level=1 → outcome = +0.0100 (+100 pips)
        state_level=2 → outcome = -0.0100 (-100 pips)

    Proves: Model can read a single categorical input and map it to output.

    Args:
        output_dir: Working directory for output
        n_sessions: Number of sessions to generate (default 1000)
        timeframe: Bar timeframe (determines bars per session)
        adr_value: ADR value for all sessions (default 1.0 so normalized = raw)
        target_outcome: Which outcome column to populate (max_up, max_down, next, sess)
        seed: Random seed

    Returns:
        Path to generated parquet file
    """
    seq_len = get_bars_per_session(timeframe)
    print(f"Generating Sanity Check data: {n_sessions} sessions x {seq_len} bars ({timeframe})...")

    np.random.seed(seed)
    total_rows = n_sessions * seq_len

    # Create base schema with ADR and timeframe
    df = create_base_schema(total_rows, adr_value, timeframe, seed)

    # Generate test pattern: level 1 or 2, deterministic outcome
    levels = np.random.choice([1, 2], size=total_rows)
    outcomes = np.where(levels == 1, 0.0100, -0.0100)  # +100 or -100 pips

    # Random events and directions (not used in sanity test pattern)
    events = np.random.choice(["SPAWN", "EXTENSION", "REVERSAL"], size=total_rows)
    directions = np.where(outcomes > 0, "UP", "DOWN")

    # Calculate normalized outcomes (with ADR=1.0, these equal raw outcomes)
    outcomes_norm = outcomes / adr_value

    # Update ALL session columns with test pattern
    for prefix in SESSION_PREFIXES:
        df = df.with_columns([
            pl.Series(f"{prefix}_state_level", levels.astype(np.int32)),
            pl.Series(f"{prefix}_state_event", events),
            pl.Series(f"{prefix}_state_dir", directions),
            pl.Series(f"{prefix}_out_{target_outcome}", outcomes),
            pl.Series(f"{prefix}_out_{target_outcome}_norm", outcomes_norm),
        ])

    # Save to parquet
    output_path = output_dir / "lstm" / "bridged"
    output_path.mkdir(parents=True, exist_ok=True)

    parquet_file = output_path / "synthetic_test_sanity.parquet"
    df.write_parquet(parquet_file, compression="zstd")

    print(f"  -> Saved {total_rows:,} rows to {parquet_file}")
    print(f"  -> Target: {target_outcome} (normalized by ADR={adr_value})")
    print(f"  -> Pattern: level=1 -> +0.01 (UP), level=2 -> -0.01 (DOWN)")
    print(f"  -> All sessions populated: {', '.join(SESSION_PREFIXES)}")

    return parquet_file


def generate_memory_test(
    output_dir: Path,
    n_sessions: int = 2000,
    timeframe: str = "M5",
    adr_value: float = 1.0,
    target_outcome: str = "max_up",
    seed: int = 42,
) -> Path:
    """
    Generate Test 2: Memory Test (Attention Verification).

    Rule:
        Outcome depends ONLY on bar 0 of each session.
        Bars 1-N contain random noise (levels 3, 4).

        Bar 0 level=1 → outcome = +0.0200 (+200 pips)
        Bar 0 level=2 → outcome = -0.0200 (-200 pips)

    Proves: Model uses attention to look back to sequence start.

    Args:
        output_dir: Working directory for output
        n_sessions: Number of sessions to generate (default 2000)
        timeframe: Bar timeframe (determines bars per session)
        adr_value: ADR value for all sessions (default 1.0 so normalized = raw)
        target_outcome: Which outcome column to populate (max_up, max_down, next, sess)
        seed: Random seed

    Returns:
        Path to generated parquet file
    """
    seq_len = get_bars_per_session(timeframe)
    print(f"Generating Memory Test data: {n_sessions} sessions x {seq_len} bars ({timeframe})...")

    np.random.seed(seed)
    total_rows = n_sessions * seq_len

    # Create base schema with ADR and timeframe
    df = create_base_schema(total_rows, adr_value, timeframe, seed)

    # Per-session "secret key" (only at bar 0)
    session_keys = np.random.choice([1, 2], size=n_sessions)
    session_outcomes = np.where(session_keys == 1, 0.0200, -0.0200)

    # Create level array: noise everywhere (levels 3-4 to force ignoring)
    levels = np.random.choice([3, 4], size=total_rows)

    # Plant the key at bar 0 of each session
    start_indices = np.arange(0, total_rows, seq_len)
    levels[start_indices] = session_keys

    # Expand outcomes to all bars (same outcome for entire session)
    outcomes = np.repeat(session_outcomes, seq_len)

    # Random events (not used in memory test pattern)
    events = np.random.choice(["SPAWN", "EXTENSION", "REVERSAL"], size=total_rows)
    directions = np.where(outcomes > 0, "UP", "DOWN")

    # Calculate normalized outcomes (with ADR=1.0, these equal raw outcomes)
    outcomes_norm = outcomes / adr_value

    # Update ALL session columns with test pattern
    for prefix in SESSION_PREFIXES:
        df = df.with_columns([
            pl.Series(f"{prefix}_state_level", levels.astype(np.int32)),
            pl.Series(f"{prefix}_state_event", events),
            pl.Series(f"{prefix}_state_dir", directions),
            pl.Series(f"{prefix}_out_{target_outcome}", outcomes),
            pl.Series(f"{prefix}_out_{target_outcome}_norm", outcomes_norm),
        ])

    # Save to parquet
    output_path = output_dir / "lstm" / "bridged"
    output_path.mkdir(parents=True, exist_ok=True)

    parquet_file = output_path / "synthetic_test_memory.parquet"
    df.write_parquet(parquet_file, compression="zstd")

    print(f"  -> Saved {total_rows:,} rows to {parquet_file}")
    print(f"  -> Target: {target_outcome} (normalized by ADR={adr_value})")
    print(f"  -> Pattern: bar[0].level=1 -> +0.02, bar[0].level=2 -> -0.02")
    print(f"  -> Bars 1-{seq_len-1} contain noise (levels 3-4)")
    print(f"  -> All sessions populated: {', '.join(SESSION_PREFIXES)}")

    return parquet_file


def generate_logic_test(
    output_dir: Path,
    n_sessions: int = 5000,
    timeframe: str = "M5",
    adr_value: float = 1.0,
    target_outcome: str = "max_up",
    seed: int = 42,
) -> Path:
    """
    Generate Test 3: Logic Test (Feature Combination).

    Rule:
        Level=1 AND Event="SPAWN" → outcome = +0.0100 (UP)
        Any other combination → outcome = -0.0100 (DOWN)

    Proves: Model can combine multiple categorical inputs.

    Args:
        output_dir: Working directory for output
        n_sessions: Number of sessions to generate (default 5000)
        timeframe: Bar timeframe (determines bars per session)
        adr_value: ADR value for all sessions (default 1.0 so normalized = raw)
        target_outcome: Which outcome column to populate (max_up, max_down, next, sess)
        seed: Random seed

    Returns:
        Path to generated parquet file
    """
    seq_len = get_bars_per_session(timeframe)
    print(f"Generating Logic Test data: {n_sessions} sessions x {seq_len} bars ({timeframe})...")

    np.random.seed(seed)
    total_rows = n_sessions * seq_len

    # Create base schema with ADR and timeframe
    df = create_base_schema(total_rows, adr_value, timeframe, seed)

    # Generate random levels (1-4) and events
    levels = np.random.choice([1, 2, 3, 4], size=total_rows)
    events = np.random.choice(["SPAWN", "EXTENSION", "REVERSAL"], size=total_rows)

    # Logic rule: Level=1 AND Event="SPAWN" -> UP, else DOWN
    is_up = (levels == 1) & (events == "SPAWN")
    outcomes = np.where(is_up, 0.0100, -0.0100)
    directions = np.where(is_up, "UP", "DOWN")

    # Calculate normalized outcomes (with ADR=1.0, these equal raw outcomes)
    outcomes_norm = outcomes / adr_value

    # Update ALL session columns with test pattern
    for prefix in SESSION_PREFIXES:
        df = df.with_columns([
            pl.Series(f"{prefix}_state_level", levels.astype(np.int32)),
            pl.Series(f"{prefix}_state_event", events),
            pl.Series(f"{prefix}_state_dir", directions),
            pl.Series(f"{prefix}_out_{target_outcome}", outcomes),
            pl.Series(f"{prefix}_out_{target_outcome}_norm", outcomes_norm),
        ])

    # Save to parquet
    output_path = output_dir / "lstm" / "bridged"
    output_path.mkdir(parents=True, exist_ok=True)

    parquet_file = output_path / "synthetic_test_logic.parquet"
    df.write_parquet(parquet_file, compression="zstd")

    # Calculate statistics
    up_count = np.sum(is_up)
    down_count = total_rows - up_count
    up_pct = 100.0 * up_count / total_rows

    print(f"  -> Saved {total_rows:,} rows to {parquet_file}")
    print(f"  -> Target: {target_outcome} (normalized by ADR={adr_value})")
    print(f"  -> Pattern: Level=1 AND Event=SPAWN -> +0.01 (UP), else -0.01 (DOWN)")
    print(f"  -> Distribution: {up_count:,} UP ({up_pct:.1f}%), {down_count:,} DOWN ({100-up_pct:.1f}%)")
    print(f"  -> All sessions populated: {', '.join(SESSION_PREFIXES)}")

    return parquet_file


# ============================================================================
# NEW: Generate from source parquet (real data based)
# ============================================================================


def is_bar_in_session(timestamp: datetime, session: str) -> bool:
    """Check if a bar timestamp falls within the given session time window."""
    hour = timestamp.hour
    start_hour, end_hour = SESSION_TIME_WINDOWS.get(session, (0, 24))
    return start_hour <= hour < end_hour


def group_bars_into_sessions(
    df: pl.DataFrame,
    session: str,
    timeframe: str,
) -> list[tuple[list[int], list[datetime]]]:
    """
    Group bars into sessions based on the target session's time window.

    Returns:
        List of (indices, timestamps) tuples, one per session day
    """
    # Add hour and date columns
    df = df.with_row_index("_idx")
    df = df.with_columns([
        pl.col("timestamp").dt.hour().alias("_hour"),
        pl.col("timestamp").dt.date().alias("_date"),
    ])

    start_hour, end_hour = SESSION_TIME_WINDOWS.get(session, (0, 24))

    # Filter to session time window
    session_df = df.filter(
        (pl.col("_hour") >= start_hour) & (pl.col("_hour") < end_hour)
    )

    # Group by date
    sessions = []
    dates = session_df.select("_date").unique().sort("_date")["_date"].to_list()

    for date in dates:
        day_df = session_df.filter(pl.col("_date") == date)
        indices = day_df["_idx"].to_list()
        timestamps = day_df["timestamp"].to_list()
        if len(indices) > 0:
            sessions.append((indices, timestamps))

    return sessions


def generate_from_source_sanity(
    source_parquet: Path,
    output_dir: Path,
    target_session: str = "lon",
    target_outcome: str = "max_up",
    timeframe: str = "M5",
    seed: int = 42,
) -> Path:
    """
    Generate Sanity Check test data by cloning an enriched parquet.

    Clones the entire enriched parquet structure (preserving volume, ADR values,
    column order) and only modifies the target session's state/outcome columns.

    Rule:
        Session A: state_level=1 (all bars) -> outcome = +0.0100
        Session B: state_level=2 (all bars) -> outcome = -0.0100
        Alternates between sessions.

    Args:
        source_parquet: Path to source ENRICHED parquet file
        output_dir: Working directory for output
        target_session: Session to generate data for (lon, ny, asia, etc.)
        target_outcome: Which outcome column to populate (max_up, max_down, next, sess)
        timeframe: Timeframe (for output filename and session grouping)
        seed: Random seed

    Returns:
        Path to generated parquet file
    """
    print(f"Generating Sanity Check from source: {source_parquet.name}")
    print(f"  Target session: {target_session}, Target outcome: {target_outcome}, Timeframe: {timeframe}")

    np.random.seed(seed)

    # Load enriched parquet - has all columns (volume, real ADR, state columns)
    df = pl.read_parquet(source_parquet)
    total_rows = len(df)
    print(f"  Loaded {total_rows:,} rows from enriched source")

    # Clone entire dataframe - preserves volume, real ADR values, column order
    result_df = df.clone()

    # Get session prefix for target session
    prefix_map = {
        "asia": "asia", "london": "lon", "lon": "lon",
        "ny": "ny", "full_day": "day", "day": "day",
        "asia_london": "asialon", "asialon": "asialon",
        "london_ny": "lonny", "lonny": "lonny",
    }
    prefix = prefix_map.get(target_session, target_session)

    # Force ADR = 1.0 for simple validation math
    adr_col = SESSION_ADR_COLS.get(prefix, f"adr_{prefix}")
    default_adr = 1.0
    print(f"  Using ADR value: {default_adr} (forced for validation)")

    # Set ADR column to 1.0
    if adr_col in result_df.columns:
        result_df = result_df.with_columns([
            pl.lit(1.0).cast(pl.Float64).alias(adr_col)
        ])

    # Reset ONLY the target session's state/outcome columns to None
    result_df = result_df.with_columns([
        pl.lit(None).cast(pl.Int32).alias(f"{prefix}_state_level"),
        pl.lit(None).cast(pl.Utf8).alias(f"{prefix}_state_event"),
        pl.lit(None).cast(pl.Utf8).alias(f"{prefix}_state_dir"),
        # Raw outcomes
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_next"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_sess"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_max_up"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_max_down"),
        # Normalized outcomes
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_next_norm"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_sess_norm"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_max_up_norm"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_max_down_norm"),
    ])

    # Group bars into sessions
    sessions = group_bars_into_sessions(df, target_session, timeframe)
    print(f"  Found {len(sessions)} sessions in target window")

    # Build update arrays initialized to None - only session rows get values
    levels = [None] * total_rows
    events = [None] * total_rows
    directions = [None] * total_rows
    outcomes = [None] * total_rows
    outcomes_norm = [None] * total_rows

    # Only fill values for rows within the session time window
    session_rows_count = 0
    for session_idx, (indices, _) in enumerate(sessions):
        is_session_a = (session_idx % 2 == 0)
        session_rows_count += len(indices)

        for idx in indices:
            if is_session_a:
                levels[idx] = 1
                outcomes[idx] = 0.0100
                outcomes_norm[idx] = 0.0100 / default_adr
                directions[idx] = "UP"
            else:
                levels[idx] = 2
                outcomes[idx] = -0.0100
                outcomes_norm[idx] = -0.0100 / default_adr
                directions[idx] = "DOWN"
            events[idx] = "EXTENSION"

    # Update ONLY the target session columns (others remain None)
    result_df = result_df.with_columns([
        pl.Series(f"{prefix}_state_level", levels).cast(pl.Int32),
        pl.Series(f"{prefix}_state_event", events).cast(pl.Utf8),
        pl.Series(f"{prefix}_state_dir", directions).cast(pl.Utf8),
        pl.Series(f"{prefix}_out_{target_outcome}", outcomes).cast(pl.Float64),
        pl.Series(f"{prefix}_out_{target_outcome}_norm", outcomes_norm).cast(pl.Float64),
    ])

    # Save output
    output_path = output_dir / "lstm" / "bridged"
    output_path.mkdir(parents=True, exist_ok=True)

    parquet_file = output_path / f"test_sanity_{timeframe}.parquet"
    result_df.write_parquet(parquet_file, compression="zstd")

    session_a_count = len([s for i, s in enumerate(sessions) if i % 2 == 0])
    session_b_count = len(sessions) - session_a_count

    print(f"  -> Saved {total_rows:,} rows to {parquet_file}")
    print(f"  -> Target: {target_outcome} (normalized by ADR={default_adr})")
    print(f"  -> Session rows: {session_rows_count:,} / {total_rows:,} ({100*session_rows_count/total_rows:.1f}%)")
    print(f"  -> Pattern: level=1 -> +0.01 (Session A), level=2 -> -0.01 (Session B)")
    print(f"  -> Distribution: {session_a_count} Session A, {session_b_count} Session B")

    return parquet_file


def generate_from_source_memory(
    source_parquet: Path,
    output_dir: Path,
    target_session: str = "lon",
    target_outcome: str = "max_up",
    timeframe: str = "M5",
    seed: int = 42,
) -> Path:
    """
    Generate Memory Test data by cloning an enriched parquet.

    Clones the entire enriched parquet structure (preserving volume, ADR values,
    column order) and only modifies the target session's state/outcome columns.

    Rule:
        Bar 0 (The Key): Randomly set state_level to 1 or 2
        Bars 1-End (The Noise): Set state_level to random levels (3, 4)
        Target:
            If Bar 0 was 1 -> outcome = +0.0100 (for entire session)
            If Bar 0 was 2 -> outcome = -0.0100 (for entire session)

    Args:
        source_parquet: Path to source ENRICHED parquet file
        output_dir: Working directory for output
        target_session: Session to generate data for
        target_outcome: Which outcome column to populate (max_up, max_down, next, sess)
        timeframe: Timeframe (for output filename and session grouping)
        seed: Random seed

    Returns:
        Path to generated parquet file
    """
    print(f"Generating Memory Test from source: {source_parquet.name}")
    print(f"  Target session: {target_session}, Target outcome: {target_outcome}, Timeframe: {timeframe}")

    np.random.seed(seed)

    # Load enriched parquet - has all columns (volume, real ADR, state columns)
    df = pl.read_parquet(source_parquet)
    total_rows = len(df)
    print(f"  Loaded {total_rows:,} rows from enriched source")

    # Clone entire dataframe - preserves volume, real ADR values, column order
    result_df = df.clone()

    # Get session prefix for target session
    prefix_map = {
        "asia": "asia", "london": "lon", "lon": "lon",
        "ny": "ny", "full_day": "day", "day": "day",
        "asia_london": "asialon", "asialon": "asialon",
        "london_ny": "lonny", "lonny": "lonny",
    }
    prefix = prefix_map.get(target_session, target_session)

    # Force ADR = 1.0 for simple validation math
    adr_col = SESSION_ADR_COLS.get(prefix, f"adr_{prefix}")
    default_adr = 1.0
    print(f"  Using ADR value: {default_adr} (forced for validation)")

    # Set ADR column to 1.0
    if adr_col in result_df.columns:
        result_df = result_df.with_columns([
            pl.lit(1.0).cast(pl.Float64).alias(adr_col)
        ])

    # Reset ONLY the target session's state/outcome columns to None
    result_df = result_df.with_columns([
        pl.lit(None).cast(pl.Int32).alias(f"{prefix}_state_level"),
        pl.lit(None).cast(pl.Utf8).alias(f"{prefix}_state_event"),
        pl.lit(None).cast(pl.Utf8).alias(f"{prefix}_state_dir"),
        # Raw outcomes
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_next"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_sess"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_max_up"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_max_down"),
        # Normalized outcomes
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_next_norm"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_sess_norm"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_max_up_norm"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_max_down_norm"),
    ])

    # Group bars into sessions
    sessions = group_bars_into_sessions(df, target_session, timeframe)
    print(f"  Found {len(sessions)} sessions in target window")

    # Build update arrays initialized to None - only session rows get values
    levels = [None] * total_rows
    events = [None] * total_rows
    directions = [None] * total_rows
    outcomes = [None] * total_rows
    outcomes_norm = [None] * total_rows

    # Generate session keys (1 or 2 for each session)
    session_keys = np.random.choice([1, 2], size=len(sessions))

    key_1_count = 0
    key_2_count = 0
    session_rows_count = 0

    for session_idx, (indices, _) in enumerate(sessions):
        if len(indices) == 0:
            continue

        session_rows_count += len(indices)
        key = session_keys[session_idx]
        outcome = 0.0100 if key == 1 else -0.0100
        outcome_norm = outcome / default_adr
        direction = "UP" if key == 1 else "DOWN"

        if key == 1:
            key_1_count += 1
        else:
            key_2_count += 1

        # Set bar 0 with the key, other bars get noise levels (3-4)
        first_idx = indices[0]
        levels[first_idx] = key

        # Set all bars in session with noise levels (except bar 0), outcome, direction
        for i, idx in enumerate(indices):
            if i > 0:  # Bars 1-N get noise levels
                levels[idx] = np.random.choice([3, 4])
            outcomes[idx] = outcome
            outcomes_norm[idx] = outcome_norm
            directions[idx] = direction
            events[idx] = "EXTENSION"

    # Update ONLY the target session columns (others remain None)
    result_df = result_df.with_columns([
        pl.Series(f"{prefix}_state_level", levels).cast(pl.Int32),
        pl.Series(f"{prefix}_state_event", events).cast(pl.Utf8),
        pl.Series(f"{prefix}_state_dir", directions).cast(pl.Utf8),
        pl.Series(f"{prefix}_out_{target_outcome}", outcomes).cast(pl.Float64),
        pl.Series(f"{prefix}_out_{target_outcome}_norm", outcomes_norm).cast(pl.Float64),
    ])

    # Save output
    output_path = output_dir / "lstm" / "bridged"
    output_path.mkdir(parents=True, exist_ok=True)

    parquet_file = output_path / f"test_memory_{timeframe}.parquet"
    result_df.write_parquet(parquet_file, compression="zstd")

    print(f"  -> Saved {total_rows:,} rows to {parquet_file}")
    print(f"  -> Target: {target_outcome} (normalized by ADR={default_adr})")
    print(f"  -> Session rows: {session_rows_count:,} / {total_rows:,} ({100*session_rows_count/total_rows:.1f}%)")
    print(f"  -> Pattern: bar[0].level=1 -> +0.01, bar[0].level=2 -> -0.01")
    print(f"  -> Bars 1-N contain noise (levels 3-4)")
    print(f"  -> Distribution: {key_1_count} key=1 sessions, {key_2_count} key=2 sessions")

    return parquet_file


def generate_from_source_logic(
    source_parquet: Path,
    output_dir: Path,
    target_session: str = "lon",
    target_outcome: str = "max_up",
    timeframe: str = "M5",
    seed: int = 42,
) -> Path:
    """
    Generate Logic Test data by cloning an enriched parquet.

    Clones the entire enriched parquet structure (preserving volume, ADR values,
    column order) and only modifies the target session's state/outcome columns.

    Rule:
        Target UP (+0.0100): state_level=1 AND state_event="SPAWN"
        Target DOWN (-0.0100): All other combinations:
            - Level 1 + "EXTENSION"
            - Level 2 + "SPAWN"
            - Level 2 + "EXTENSION"
        Distribution: Randomly mix these 4 conditions across sessions.

    Args:
        source_parquet: Path to source ENRICHED parquet file
        output_dir: Working directory for output
        target_session: Session to generate data for
        target_outcome: Which outcome column to populate (max_up, max_down, next, sess)
        timeframe: Timeframe (for output filename and session grouping)
        seed: Random seed

    Returns:
        Path to generated parquet file
    """
    print(f"Generating Logic Test from source: {source_parquet.name}")
    print(f"  Target session: {target_session}, Target outcome: {target_outcome}, Timeframe: {timeframe}")

    np.random.seed(seed)

    # Load enriched parquet - has all columns (volume, real ADR, state columns)
    df = pl.read_parquet(source_parquet)
    total_rows = len(df)
    print(f"  Loaded {total_rows:,} rows from enriched source")

    # Clone entire dataframe - preserves volume, real ADR values, column order
    result_df = df.clone()

    # Get session prefix for target session
    prefix_map = {
        "asia": "asia", "london": "lon", "lon": "lon",
        "ny": "ny", "full_day": "day", "day": "day",
        "asia_london": "asialon", "asialon": "asialon",
        "london_ny": "lonny", "lonny": "lonny",
    }
    prefix = prefix_map.get(target_session, target_session)

    # Force ADR = 1.0 for simple validation math
    adr_col = SESSION_ADR_COLS.get(prefix, f"adr_{prefix}")
    default_adr = 1.0
    print(f"  Using ADR value: {default_adr} (forced for validation)")

    # Set ADR column to 1.0
    if adr_col in result_df.columns:
        result_df = result_df.with_columns([
            pl.lit(1.0).cast(pl.Float64).alias(adr_col)
        ])

    # Reset ONLY the target session's state/outcome columns to None
    result_df = result_df.with_columns([
        pl.lit(None).cast(pl.Int32).alias(f"{prefix}_state_level"),
        pl.lit(None).cast(pl.Utf8).alias(f"{prefix}_state_event"),
        pl.lit(None).cast(pl.Utf8).alias(f"{prefix}_state_dir"),
        # Raw outcomes
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_next"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_sess"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_max_up"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_max_down"),
        # Normalized outcomes
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_next_norm"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_sess_norm"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_max_up_norm"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_max_down_norm"),
    ])

    # Group bars into sessions
    sessions = group_bars_into_sessions(df, target_session, timeframe)
    print(f"  Found {len(sessions)} sessions in target window")

    # Define the 4 conditions
    CONDITIONS = [
        (1, "SPAWN"),      # UP condition
        (1, "EXTENSION"),  # DOWN condition
        (2, "SPAWN"),      # DOWN condition
        (2, "EXTENSION"),  # DOWN condition
    ]

    # Build update arrays - initialize with None, only fill session rows
    levels = [None] * total_rows
    events = [None] * total_rows
    directions = [None] * total_rows
    outcomes = [None] * total_rows
    outcomes_norm = [None] * total_rows

    # Randomly assign conditions to sessions
    condition_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    for session_idx, (indices, _) in enumerate(sessions):
        if len(indices) == 0:
            continue

        # Pick a random condition for this session
        cond_idx = np.random.randint(0, 4)
        level, event = CONDITIONS[cond_idx]
        condition_counts[cond_idx] += 1

        # Determine outcome based on condition
        is_up = (level == 1 and event == "SPAWN")
        outcome = 0.0100 if is_up else -0.0100
        outcome_norm = outcome / default_adr
        direction = "UP" if is_up else "DOWN"

        # Apply to all bars in this session
        for idx in indices:
            levels[idx] = level
            events[idx] = event
            directions[idx] = direction
            outcomes[idx] = outcome
            outcomes_norm[idx] = outcome_norm

    # Count session rows for logging
    session_rows_count = sum(len(indices) for indices, _ in sessions)

    # Update ONLY the target session columns (others remain None)
    result_df = result_df.with_columns([
        pl.Series(f"{prefix}_state_level", levels).cast(pl.Int32),
        pl.Series(f"{prefix}_state_event", events).cast(pl.Utf8),
        pl.Series(f"{prefix}_state_dir", directions).cast(pl.Utf8),
        pl.Series(f"{prefix}_out_{target_outcome}", outcomes).cast(pl.Float64),
        pl.Series(f"{prefix}_out_{target_outcome}_norm", outcomes_norm).cast(pl.Float64),
    ])

    # Save output
    output_path = output_dir / "lstm" / "bridged"
    output_path.mkdir(parents=True, exist_ok=True)

    parquet_file = output_path / f"test_logic_{timeframe}.parquet"
    result_df.write_parquet(parquet_file, compression="zstd")

    up_sessions = condition_counts[0]  # Level 1 + SPAWN
    down_sessions = sum(condition_counts[i] for i in [1, 2, 3])

    print(f"  -> Saved {total_rows:,} rows to {parquet_file}")
    print(f"  -> Target: {target_outcome} (normalized by ADR={default_adr})")
    print(f"  -> Session rows: {session_rows_count:,} / {total_rows:,} ({100*session_rows_count/total_rows:.1f}%)")
    print(f"  -> Pattern: Level=1 AND Event=SPAWN -> +0.01 (UP), else -0.01 (DOWN)")
    print(f"  -> Condition distribution:")
    print(f"      L1+SPAWN (UP):     {condition_counts[0]} sessions")
    print(f"      L1+EXTENSION (DN): {condition_counts[1]} sessions")
    print(f"      L2+SPAWN (DN):     {condition_counts[2]} sessions")
    print(f"      L2+EXTENSION (DN): {condition_counts[3]} sessions")
    print(f"  -> Total: {up_sessions} UP, {down_sessions} DOWN")

    return parquet_file


# ============================================================================
# NEW: From-Source Target-Specific Validation Tests
# ============================================================================


def generate_from_source_next(
    source_parquet: Path,
    output_dir: Path,
    target_session: str = "lon",
    target_outcome: str = "next",
    timeframe: str = "M5",
    seed: int = 42,
) -> Path:
    """
    Generate Next Bar test data by cloning an enriched parquet.

    Tests if the model can predict the immediate next move (out_next).

    Rule:
        state_dir = UP   -> out_next = +0.5 (ADR normalized)
        state_dir = DOWN -> out_next = -0.5 (ADR normalized)

    Adversarial Trap:
        - out_sess is set to OPPOSITE direction (if next is +0.5, sess is -2.0)
        - out_max_up/out_max_down set to misleading values
        - If model cheats by looking at other outcomes, it will fail

    Args:
        source_parquet: Path to source ENRICHED parquet file
        output_dir: Working directory for output
        target_session: Session to generate data for
        target_outcome: Should be "next" for this test
        timeframe: Timeframe (for output filename and session grouping)
        seed: Random seed

    Returns:
        Path to generated parquet file
    """
    print(f"Generating Next Bar Test from source: {source_parquet.name}")
    print(f"  Target session: {target_session}, Target outcome: {target_outcome}, Timeframe: {timeframe}")

    np.random.seed(seed)

    # Load enriched parquet
    df = pl.read_parquet(source_parquet)
    total_rows = len(df)
    print(f"  Loaded {total_rows:,} rows from enriched source")

    # Clone entire dataframe
    result_df = df.clone()

    # Get session prefix
    prefix_map = {
        "asia": "asia", "london": "lon", "lon": "lon",
        "ny": "ny", "full_day": "day", "day": "day",
        "asia_london": "asialon", "asialon": "asialon",
        "london_ny": "lonny", "lonny": "lonny",
    }
    prefix = prefix_map.get(target_session, target_session)

    # Force ADR = 1.0 for simple validation math
    adr_col = SESSION_ADR_COLS.get(prefix, f"adr_{prefix}")
    default_adr = 1.0
    print(f"  Using ADR value: {default_adr} (forced for validation)")

    # Set ADR column to 1.0
    if adr_col in result_df.columns:
        result_df = result_df.with_columns([
            pl.lit(1.0).cast(pl.Float64).alias(adr_col)
        ])

    # Reset target session's state/outcome columns to None
    result_df = result_df.with_columns([
        pl.lit(None).cast(pl.Int32).alias(f"{prefix}_state_level"),
        pl.lit(None).cast(pl.Utf8).alias(f"{prefix}_state_event"),
        pl.lit(None).cast(pl.Utf8).alias(f"{prefix}_state_dir"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_next"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_sess"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_max_up"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_max_down"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_next_norm"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_sess_norm"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_max_up_norm"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_max_down_norm"),
    ])

    # Group bars into sessions
    sessions = group_bars_into_sessions(df, target_session, timeframe)
    print(f"  Found {len(sessions)} sessions in target window")

    # Build update arrays
    levels = [None] * total_rows
    events = [None] * total_rows
    directions = [None] * total_rows
    out_next = [None] * total_rows
    out_next_norm = [None] * total_rows
    out_sess = [None] * total_rows
    out_sess_norm = [None] * total_rows
    out_max_up = [None] * total_rows
    out_max_up_norm = [None] * total_rows
    out_max_down = [None] * total_rows
    out_max_down_norm = [None] * total_rows

    up_count = 0
    down_count = 0
    session_rows_count = 0

    for session_idx, (indices, _) in enumerate(sessions):
        if len(indices) == 0:
            continue

        session_rows_count += len(indices)

        for idx in indices:
            # Random direction for each bar
            is_up = np.random.random() > 0.5
            if is_up:
                up_count += 1
                directions[idx] = "UP"
                # Target: out_next = +0.5
                out_next[idx] = 0.5
                out_next_norm[idx] = 0.5 / default_adr
                # ADVERSARIAL TRAP: out_sess is OPPOSITE
                out_sess[idx] = -2.0
                out_sess_norm[idx] = -2.0 / default_adr
                # More traps
                out_max_up[idx] = -1.0
                out_max_up_norm[idx] = -1.0 / default_adr
                out_max_down[idx] = -3.0
                out_max_down_norm[idx] = -3.0 / default_adr
            else:
                down_count += 1
                directions[idx] = "DOWN"
                # Target: out_next = -0.5
                out_next[idx] = -0.5
                out_next_norm[idx] = -0.5 / default_adr
                # ADVERSARIAL TRAP: out_sess is OPPOSITE
                out_sess[idx] = 2.0
                out_sess_norm[idx] = 2.0 / default_adr
                # More traps
                out_max_up[idx] = 3.0
                out_max_up_norm[idx] = 3.0 / default_adr
                out_max_down[idx] = 0.0
                out_max_down_norm[idx] = 0.0

            # Random level and event (not used in pattern)
            levels[idx] = np.random.choice([1, 2, 3, 4])
            events[idx] = np.random.choice(["SPAWN", "EXTENSION", "REVERSAL"])

    # Update columns
    result_df = result_df.with_columns([
        pl.Series(f"{prefix}_state_level", levels).cast(pl.Int32),
        pl.Series(f"{prefix}_state_event", events).cast(pl.Utf8),
        pl.Series(f"{prefix}_state_dir", directions).cast(pl.Utf8),
        pl.Series(f"{prefix}_out_next", out_next).cast(pl.Float64),
        pl.Series(f"{prefix}_out_next_norm", out_next_norm).cast(pl.Float64),
        pl.Series(f"{prefix}_out_sess", out_sess).cast(pl.Float64),
        pl.Series(f"{prefix}_out_sess_norm", out_sess_norm).cast(pl.Float64),
        pl.Series(f"{prefix}_out_max_up", out_max_up).cast(pl.Float64),
        pl.Series(f"{prefix}_out_max_up_norm", out_max_up_norm).cast(pl.Float64),
        pl.Series(f"{prefix}_out_max_down", out_max_down).cast(pl.Float64),
        pl.Series(f"{prefix}_out_max_down_norm", out_max_down_norm).cast(pl.Float64),
    ])

    # Save output
    output_path = output_dir / "lstm" / "bridged"
    output_path.mkdir(parents=True, exist_ok=True)

    parquet_file = output_path / f"test_next_{timeframe}.parquet"
    result_df.write_parquet(parquet_file, compression="zstd")

    print(f"  -> Saved {total_rows:,} rows to {parquet_file}")
    print(f"  -> Target: out_next (normalized by ADR={default_adr})")
    print(f"  -> Session rows: {session_rows_count:,} / {total_rows:,} ({100*session_rows_count/total_rows:.1f}%)")
    print(f"  -> Pattern: state_dir=UP -> +0.5, state_dir=DOWN -> -0.5")
    print(f"  -> Adversarial: out_sess set to OPPOSITE direction")
    print(f"  -> Distribution: {up_count:,} UP, {down_count:,} DOWN")

    return parquet_file


def generate_from_source_next5(
    source_parquet: Path,
    output_dir: Path,
    target_session: str = "lon",
    target_outcome: str = "next5",
    timeframe: str = "M5",
    seed: int = 42,
) -> Path:
    """
    Generate Next 5 Bars test data by cloning an enriched parquet.

    Tests if the model can predict movement 5 bars ahead (out_next5) while
    ignoring immediate 1-bar noise.

    Rule:
        state_dir = UP   -> out_next5 = +2.0 (ADR normalized)
        state_dir = DOWN -> out_next5 = -2.0 (ADR normalized)

    Adversarial Trap:
        - out_next (1-bar) is set to OPPOSITE direction
        - If model cheats by looking at next-bar outcome, it will fail
        - This proves the model looks 5 bars ahead and ignores immediate noise

    Args:
        source_parquet: Path to source ENRICHED parquet file
        output_dir: Working directory for output
        target_session: Session to generate data for
        target_outcome: Should be "next5" for this test
        timeframe: Timeframe (for output filename and session grouping)
        seed: Random seed

    Returns:
        Path to generated parquet file
    """
    print(f"Generating Next 5 Bars Test from source: {source_parquet.name}")
    print(f"  Target session: {target_session}, Target outcome: {target_outcome}, Timeframe: {timeframe}")

    np.random.seed(seed)

    # Load enriched parquet
    df = pl.read_parquet(source_parquet)
    total_rows = len(df)
    print(f"  Loaded {total_rows:,} rows from enriched source")

    # Clone entire dataframe
    result_df = df.clone()

    # Get session prefix
    prefix_map = {
        "asia": "asia", "london": "lon", "lon": "lon",
        "ny": "ny", "full_day": "day", "day": "day",
        "asia_london": "asialon", "asialon": "asialon",
        "london_ny": "lonny", "lonny": "lonny",
    }
    prefix = prefix_map.get(target_session, target_session)

    # Force ADR = 1.0 for simple validation math
    adr_col = SESSION_ADR_COLS.get(prefix, f"adr_{prefix}")
    default_adr = 1.0
    print(f"  Using ADR value: {default_adr} (forced for validation)")

    # Set ADR column to 1.0
    if adr_col in result_df.columns:
        result_df = result_df.with_columns([
            pl.lit(1.0).cast(pl.Float64).alias(adr_col)
        ])

    # Reset target session's state/outcome columns to None
    result_df = result_df.with_columns([
        pl.lit(None).cast(pl.Int32).alias(f"{prefix}_state_level"),
        pl.lit(None).cast(pl.Utf8).alias(f"{prefix}_state_event"),
        pl.lit(None).cast(pl.Utf8).alias(f"{prefix}_state_dir"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_next"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_next5"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_sess"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_max_up"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_max_down"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_next_norm"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_next5_norm"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_sess_norm"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_max_up_norm"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_max_down_norm"),
    ])

    # Group bars into sessions
    sessions = group_bars_into_sessions(df, target_session, timeframe)
    print(f"  Found {len(sessions)} sessions in target window")

    # Build update arrays
    levels = [None] * total_rows
    events = [None] * total_rows
    directions = [None] * total_rows
    out_next = [None] * total_rows
    out_next_norm = [None] * total_rows
    out_next5 = [None] * total_rows
    out_next5_norm = [None] * total_rows
    out_sess = [None] * total_rows
    out_sess_norm = [None] * total_rows
    out_max_up = [None] * total_rows
    out_max_up_norm = [None] * total_rows
    out_max_down = [None] * total_rows
    out_max_down_norm = [None] * total_rows

    up_count = 0
    down_count = 0
    session_rows_count = 0

    for session_idx, (indices, _) in enumerate(sessions):
        if len(indices) == 0:
            continue

        session_rows_count += len(indices)

        for idx in indices:
            # Random direction for each bar
            is_up = np.random.random() > 0.5
            if is_up:
                up_count += 1
                directions[idx] = "UP"
                # Target: out_next5 = +2.0 (sustained up move)
                out_next5[idx] = 2.0
                out_next5_norm[idx] = 2.0 / default_adr
                # ADVERSARIAL TRAP: out_next (1-bar) is OPPOSITE
                out_next[idx] = -0.5
                out_next_norm[idx] = -0.5 / default_adr
                # out_sess also misleading
                out_sess[idx] = -1.0
                out_sess_norm[idx] = -1.0 / default_adr
                # More traps
                out_max_up[idx] = -1.0
                out_max_up_norm[idx] = -1.0 / default_adr
                out_max_down[idx] = -3.0
                out_max_down_norm[idx] = -3.0 / default_adr
            else:
                down_count += 1
                directions[idx] = "DOWN"
                # Target: out_next5 = -2.0 (sustained down move)
                out_next5[idx] = -2.0
                out_next5_norm[idx] = -2.0 / default_adr
                # ADVERSARIAL TRAP: out_next (1-bar) is OPPOSITE
                out_next[idx] = 0.5
                out_next_norm[idx] = 0.5 / default_adr
                # out_sess also misleading
                out_sess[idx] = 1.0
                out_sess_norm[idx] = 1.0 / default_adr
                # More traps
                out_max_up[idx] = 3.0
                out_max_up_norm[idx] = 3.0 / default_adr
                out_max_down[idx] = 0.0
                out_max_down_norm[idx] = 0.0

            # Random level and event (not used in pattern)
            levels[idx] = np.random.choice([1, 2, 3, 4])
            events[idx] = np.random.choice(["SPAWN", "EXTENSION", "REVERSAL"])

    # Update columns
    result_df = result_df.with_columns([
        pl.Series(f"{prefix}_state_level", levels).cast(pl.Int32),
        pl.Series(f"{prefix}_state_event", events).cast(pl.Utf8),
        pl.Series(f"{prefix}_state_dir", directions).cast(pl.Utf8),
        pl.Series(f"{prefix}_out_next", out_next).cast(pl.Float64),
        pl.Series(f"{prefix}_out_next_norm", out_next_norm).cast(pl.Float64),
        pl.Series(f"{prefix}_out_next5", out_next5).cast(pl.Float64),
        pl.Series(f"{prefix}_out_next5_norm", out_next5_norm).cast(pl.Float64),
        pl.Series(f"{prefix}_out_sess", out_sess).cast(pl.Float64),
        pl.Series(f"{prefix}_out_sess_norm", out_sess_norm).cast(pl.Float64),
        pl.Series(f"{prefix}_out_max_up", out_max_up).cast(pl.Float64),
        pl.Series(f"{prefix}_out_max_up_norm", out_max_up_norm).cast(pl.Float64),
        pl.Series(f"{prefix}_out_max_down", out_max_down).cast(pl.Float64),
        pl.Series(f"{prefix}_out_max_down_norm", out_max_down_norm).cast(pl.Float64),
    ])

    # Save output
    output_path = output_dir / "lstm" / "bridged"
    output_path.mkdir(parents=True, exist_ok=True)

    parquet_file = output_path / f"test_next5_{timeframe}.parquet"
    result_df.write_parquet(parquet_file, compression="zstd")

    print(f"  -> Saved {total_rows:,} rows to {parquet_file}")
    print(f"  -> Target: out_next5 (normalized by ADR={default_adr})")
    print(f"  -> Session rows: {session_rows_count:,} / {total_rows:,} ({100*session_rows_count/total_rows:.1f}%)")
    print(f"  -> Pattern: state_dir=UP -> +2.0, state_dir=DOWN -> -2.0")
    print(f"  -> Adversarial: out_next (1-bar) set to OPPOSITE direction")
    print(f"  -> Distribution: {up_count:,} UP, {down_count:,} DOWN")

    return parquet_file


def generate_from_source_close(
    source_parquet: Path,
    output_dir: Path,
    target_session: str = "lon",
    target_outcome: str = "sess",
    timeframe: str = "M5",
    seed: int = 42,
) -> Path:
    """
    Generate Session Close test data by cloning an enriched parquet.

    Tests if the model can predict the final session close (out_sess).
    The key is at Bar 0 only - forces model to use attention to look back.

    Rule:
        Bar 0 state_level = 1 -> out_sess = +2.0 (End of day is High)
        Bar 0 state_level = 2 -> out_sess = -2.0 (End of day is Low)
        Bars 1-N have noise levels (3, 4)

    Differentiation:
        - out_max_up is set to 0.0 (forces learning "where we end" not "how high")
        - out_next is random noise (not correlated with session close)

    Args:
        source_parquet: Path to source ENRICHED parquet file
        output_dir: Working directory for output
        target_session: Session to generate data for
        target_outcome: Should be "sess" for this test
        timeframe: Timeframe (for output filename and session grouping)
        seed: Random seed

    Returns:
        Path to generated parquet file
    """
    print(f"Generating Session Close Test from source: {source_parquet.name}")
    print(f"  Target session: {target_session}, Target outcome: {target_outcome}, Timeframe: {timeframe}")

    np.random.seed(seed)

    # Load enriched parquet
    df = pl.read_parquet(source_parquet)
    total_rows = len(df)
    print(f"  Loaded {total_rows:,} rows from enriched source")

    # Clone entire dataframe
    result_df = df.clone()

    # Get session prefix
    prefix_map = {
        "asia": "asia", "london": "lon", "lon": "lon",
        "ny": "ny", "full_day": "day", "day": "day",
        "asia_london": "asialon", "asialon": "asialon",
        "london_ny": "lonny", "lonny": "lonny",
    }
    prefix = prefix_map.get(target_session, target_session)

    # Force ADR = 1.0 for simple validation math
    adr_col = SESSION_ADR_COLS.get(prefix, f"adr_{prefix}")
    default_adr = 1.0
    print(f"  Using ADR value: {default_adr} (forced for validation)")

    # Set ADR column to 1.0
    if adr_col in result_df.columns:
        result_df = result_df.with_columns([
            pl.lit(1.0).cast(pl.Float64).alias(adr_col)
        ])

    # Reset target session's state/outcome columns to None
    result_df = result_df.with_columns([
        pl.lit(None).cast(pl.Int32).alias(f"{prefix}_state_level"),
        pl.lit(None).cast(pl.Utf8).alias(f"{prefix}_state_event"),
        pl.lit(None).cast(pl.Utf8).alias(f"{prefix}_state_dir"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_next"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_sess"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_max_up"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_max_down"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_next_norm"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_sess_norm"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_max_up_norm"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_max_down_norm"),
    ])

    # Group bars into sessions
    sessions = group_bars_into_sessions(df, target_session, timeframe)
    print(f"  Found {len(sessions)} sessions in target window")

    # Build update arrays
    levels = [None] * total_rows
    events = [None] * total_rows
    directions = [None] * total_rows
    out_next = [None] * total_rows
    out_next_norm = [None] * total_rows
    out_sess = [None] * total_rows
    out_sess_norm = [None] * total_rows
    out_max_up = [None] * total_rows
    out_max_up_norm = [None] * total_rows
    out_max_down = [None] * total_rows
    out_max_down_norm = [None] * total_rows

    key_1_count = 0
    key_2_count = 0
    session_rows_count = 0

    for session_idx, (indices, _) in enumerate(sessions):
        if len(indices) == 0:
            continue

        session_rows_count += len(indices)

        # Randomly choose key for this session (at Bar 0)
        key = 1 if np.random.random() > 0.5 else 2
        if key == 1:
            key_1_count += 1
            sess_outcome = 2.0
            direction = "UP"
        else:
            key_2_count += 1
            sess_outcome = -2.0
            direction = "DOWN"

        for i, idx in enumerate(indices):
            directions[idx] = direction
            events[idx] = np.random.choice(["SPAWN", "EXTENSION", "REVERSAL"])

            # Bar 0 gets the key, other bars get noise levels
            if i == 0:
                levels[idx] = key
            else:
                levels[idx] = np.random.choice([3, 4])

            # Target: out_sess is same for all bars in session
            out_sess[idx] = sess_outcome
            out_sess_norm[idx] = sess_outcome / default_adr

            # Differentiation: out_max_up is always 0.0
            out_max_up[idx] = 0.0
            out_max_up_norm[idx] = 0.0

            # Noise: out_next is random
            out_next[idx] = np.random.uniform(-0.5, 0.5)
            out_next_norm[idx] = out_next[idx] / default_adr

            # Noise: out_max_down is random
            out_max_down[idx] = np.random.uniform(-1.0, 0.0)
            out_max_down_norm[idx] = out_max_down[idx] / default_adr

    # Update columns
    result_df = result_df.with_columns([
        pl.Series(f"{prefix}_state_level", levels).cast(pl.Int32),
        pl.Series(f"{prefix}_state_event", events).cast(pl.Utf8),
        pl.Series(f"{prefix}_state_dir", directions).cast(pl.Utf8),
        pl.Series(f"{prefix}_out_next", out_next).cast(pl.Float64),
        pl.Series(f"{prefix}_out_next_norm", out_next_norm).cast(pl.Float64),
        pl.Series(f"{prefix}_out_sess", out_sess).cast(pl.Float64),
        pl.Series(f"{prefix}_out_sess_norm", out_sess_norm).cast(pl.Float64),
        pl.Series(f"{prefix}_out_max_up", out_max_up).cast(pl.Float64),
        pl.Series(f"{prefix}_out_max_up_norm", out_max_up_norm).cast(pl.Float64),
        pl.Series(f"{prefix}_out_max_down", out_max_down).cast(pl.Float64),
        pl.Series(f"{prefix}_out_max_down_norm", out_max_down_norm).cast(pl.Float64),
    ])

    # Save output
    output_path = output_dir / "lstm" / "bridged"
    output_path.mkdir(parents=True, exist_ok=True)

    parquet_file = output_path / f"test_close_{timeframe}.parquet"
    result_df.write_parquet(parquet_file, compression="zstd")

    print(f"  -> Saved {total_rows:,} rows to {parquet_file}")
    print(f"  -> Target: out_sess (normalized by ADR={default_adr})")
    print(f"  -> Session rows: {session_rows_count:,} / {total_rows:,} ({100*session_rows_count/total_rows:.1f}%)")
    print(f"  -> Pattern: bar[0].level=1 -> +2.0, bar[0].level=2 -> -2.0")
    print(f"  -> Bars 1-N have noise levels (3-4)")
    print(f"  -> Differentiation: out_max_up = 0.0")
    print(f"  -> Distribution: {key_1_count} key=1 sessions, {key_2_count} key=2 sessions")

    return parquet_file


def generate_from_source_max_up(
    source_parquet: Path,
    output_dir: Path,
    target_session: str = "lon",
    target_outcome: str = "max_up",
    timeframe: str = "M5",
    seed: int = 42,
) -> Path:
    """
    Generate Max Upside test data by cloning an enriched parquet.

    Tests if the model can predict the highest point reached (out_max_up).
    Specifically tests Level 1 + SPAWN combination.

    Rule:
        state_level = 1 AND state_event = "SPAWN":
            -> out_max_up = +5.0 (Huge wick to the upside)
            -> out_sess = 0.0 (Price crashes back to start - flat close)
        All other combinations:
            -> out_max_up = +0.5 (Small move)
            -> out_sess = random (uncorrelated)

    Why: Proves model can spot a temporary spike even if the day ends flat.

    Args:
        source_parquet: Path to source ENRICHED parquet file
        output_dir: Working directory for output
        target_session: Session to generate data for
        target_outcome: Should be "max_up" for this test
        timeframe: Timeframe (for output filename and session grouping)
        seed: Random seed

    Returns:
        Path to generated parquet file
    """
    print(f"Generating Max Up Test from source: {source_parquet.name}")
    print(f"  Target session: {target_session}, Target outcome: {target_outcome}, Timeframe: {timeframe}")

    np.random.seed(seed)

    # Load enriched parquet
    df = pl.read_parquet(source_parquet)
    total_rows = len(df)
    print(f"  Loaded {total_rows:,} rows from enriched source")

    # Clone entire dataframe
    result_df = df.clone()

    # Get session prefix
    prefix_map = {
        "asia": "asia", "london": "lon", "lon": "lon",
        "ny": "ny", "full_day": "day", "day": "day",
        "asia_london": "asialon", "asialon": "asialon",
        "london_ny": "lonny", "lonny": "lonny",
    }
    prefix = prefix_map.get(target_session, target_session)

    # Force ADR = 1.0 for simple validation math
    adr_col = SESSION_ADR_COLS.get(prefix, f"adr_{prefix}")
    default_adr = 1.0
    print(f"  Using ADR value: {default_adr} (forced for validation)")

    # Set ADR column to 1.0
    if adr_col in result_df.columns:
        result_df = result_df.with_columns([
            pl.lit(1.0).cast(pl.Float64).alias(adr_col)
        ])

    # Reset target session's state/outcome columns to None
    result_df = result_df.with_columns([
        pl.lit(None).cast(pl.Int32).alias(f"{prefix}_state_level"),
        pl.lit(None).cast(pl.Utf8).alias(f"{prefix}_state_event"),
        pl.lit(None).cast(pl.Utf8).alias(f"{prefix}_state_dir"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_next"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_sess"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_max_up"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_max_down"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_next_norm"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_sess_norm"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_max_up_norm"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_max_down_norm"),
    ])

    # Group bars into sessions
    sessions = group_bars_into_sessions(df, target_session, timeframe)
    print(f"  Found {len(sessions)} sessions in target window")

    # Build update arrays
    levels = [None] * total_rows
    events = [None] * total_rows
    directions = [None] * total_rows
    out_next = [None] * total_rows
    out_next_norm = [None] * total_rows
    out_sess = [None] * total_rows
    out_sess_norm = [None] * total_rows
    out_max_up = [None] * total_rows
    out_max_up_norm = [None] * total_rows
    out_max_down = [None] * total_rows
    out_max_down_norm = [None] * total_rows

    spike_count = 0
    normal_count = 0
    session_rows_count = 0

    for session_idx, (indices, _) in enumerate(sessions):
        if len(indices) == 0:
            continue

        session_rows_count += len(indices)

        for idx in indices:
            # Random level and event
            level = np.random.choice([1, 2, 3, 4])
            event = np.random.choice(["SPAWN", "EXTENSION", "REVERSAL"])
            levels[idx] = level
            events[idx] = event

            # Logic rule: Level=1 AND Event="SPAWN" -> big spike
            is_spike = (level == 1 and event == "SPAWN")

            if is_spike:
                spike_count += 1
                # Target: out_max_up = +5.0
                out_max_up[idx] = 5.0
                out_max_up_norm[idx] = 5.0 / default_adr
                # Key differentiation: out_sess is 0.0 (flat close despite spike)
                out_sess[idx] = 0.0
                out_sess_norm[idx] = 0.0
                directions[idx] = "UP"
            else:
                normal_count += 1
                # Target: out_max_up = +0.5
                out_max_up[idx] = 0.5
                out_max_up_norm[idx] = 0.5 / default_adr
                # out_sess is random
                out_sess[idx] = np.random.uniform(-1.0, 1.0)
                out_sess_norm[idx] = out_sess[idx] / default_adr
                directions[idx] = np.random.choice(["UP", "DOWN"])

            # Uncorrelated noise for other outcomes
            out_next[idx] = np.random.uniform(-0.5, 0.5)
            out_next_norm[idx] = out_next[idx] / default_adr
            out_max_down[idx] = np.random.uniform(-2.0, 0.0)
            out_max_down_norm[idx] = out_max_down[idx] / default_adr

    # Update columns
    result_df = result_df.with_columns([
        pl.Series(f"{prefix}_state_level", levels).cast(pl.Int32),
        pl.Series(f"{prefix}_state_event", events).cast(pl.Utf8),
        pl.Series(f"{prefix}_state_dir", directions).cast(pl.Utf8),
        pl.Series(f"{prefix}_out_next", out_next).cast(pl.Float64),
        pl.Series(f"{prefix}_out_next_norm", out_next_norm).cast(pl.Float64),
        pl.Series(f"{prefix}_out_sess", out_sess).cast(pl.Float64),
        pl.Series(f"{prefix}_out_sess_norm", out_sess_norm).cast(pl.Float64),
        pl.Series(f"{prefix}_out_max_up", out_max_up).cast(pl.Float64),
        pl.Series(f"{prefix}_out_max_up_norm", out_max_up_norm).cast(pl.Float64),
        pl.Series(f"{prefix}_out_max_down", out_max_down).cast(pl.Float64),
        pl.Series(f"{prefix}_out_max_down_norm", out_max_down_norm).cast(pl.Float64),
    ])

    # Save output
    output_path = output_dir / "lstm" / "bridged"
    output_path.mkdir(parents=True, exist_ok=True)

    parquet_file = output_path / f"test_max_up_{timeframe}.parquet"
    result_df.write_parquet(parquet_file, compression="zstd")

    spike_pct = 100.0 * spike_count / (spike_count + normal_count) if (spike_count + normal_count) > 0 else 0

    print(f"  -> Saved {total_rows:,} rows to {parquet_file}")
    print(f"  -> Target: out_max_up (normalized by ADR={default_adr})")
    print(f"  -> Session rows: {session_rows_count:,} / {total_rows:,} ({100*session_rows_count/total_rows:.1f}%)")
    print(f"  -> Pattern: Level=1 AND Event=SPAWN -> +5.0, else +0.5")
    print(f"  -> Differentiation: Spikes have out_sess=0.0 (flat close)")
    print(f"  -> Distribution: {spike_count:,} spikes ({spike_pct:.1f}%), {normal_count:,} normal")

    return parquet_file


def generate_from_source_max_down(
    source_parquet: Path,
    output_dir: Path,
    target_session: str = "lon",
    target_outcome: str = "max_down",
    timeframe: str = "M5",
    seed: int = 42,
) -> Path:
    """
    Generate Max Drawdown test data by cloning an enriched parquet.

    Tests if the model can predict drawdown risk (out_max_down).
    Level 2 indicates danger.

    Rule:
        state_level = 2 -> out_max_down = -3.0 (Deep crash)
        state_level = 1 -> out_max_down = -0.1 (Safe, minimal drawdown)
        state_level = 3,4 -> random between -0.5 and -1.5 (noise)

    Why: Ensures model specifically learns to predict *how low* price will go.

    Args:
        source_parquet: Path to source ENRICHED parquet file
        output_dir: Working directory for output
        target_session: Session to generate data for
        target_outcome: Should be "max_down" for this test
        timeframe: Timeframe (for output filename and session grouping)
        seed: Random seed

    Returns:
        Path to generated parquet file
    """
    print(f"Generating Max Down Test from source: {source_parquet.name}")
    print(f"  Target session: {target_session}, Target outcome: {target_outcome}, Timeframe: {timeframe}")

    np.random.seed(seed)

    # Load enriched parquet
    df = pl.read_parquet(source_parquet)
    total_rows = len(df)
    print(f"  Loaded {total_rows:,} rows from enriched source")

    # Clone entire dataframe
    result_df = df.clone()

    # Get session prefix
    prefix_map = {
        "asia": "asia", "london": "lon", "lon": "lon",
        "ny": "ny", "full_day": "day", "day": "day",
        "asia_london": "asialon", "asialon": "asialon",
        "london_ny": "lonny", "lonny": "lonny",
    }
    prefix = prefix_map.get(target_session, target_session)

    # Force ADR = 1.0 for simple validation math
    adr_col = SESSION_ADR_COLS.get(prefix, f"adr_{prefix}")
    default_adr = 1.0
    print(f"  Using ADR value: {default_adr} (forced for validation)")

    # Set ADR column to 1.0
    if adr_col in result_df.columns:
        result_df = result_df.with_columns([
            pl.lit(1.0).cast(pl.Float64).alias(adr_col)
        ])

    # Reset target session's state/outcome columns to None
    result_df = result_df.with_columns([
        pl.lit(None).cast(pl.Int32).alias(f"{prefix}_state_level"),
        pl.lit(None).cast(pl.Utf8).alias(f"{prefix}_state_event"),
        pl.lit(None).cast(pl.Utf8).alias(f"{prefix}_state_dir"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_next"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_sess"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_max_up"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_max_down"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_next_norm"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_sess_norm"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_max_up_norm"),
        pl.lit(None).cast(pl.Float64).alias(f"{prefix}_out_max_down_norm"),
    ])

    # Group bars into sessions
    sessions = group_bars_into_sessions(df, target_session, timeframe)
    print(f"  Found {len(sessions)} sessions in target window")

    # Build update arrays
    levels = [None] * total_rows
    events = [None] * total_rows
    directions = [None] * total_rows
    out_next = [None] * total_rows
    out_next_norm = [None] * total_rows
    out_sess = [None] * total_rows
    out_sess_norm = [None] * total_rows
    out_max_up = [None] * total_rows
    out_max_up_norm = [None] * total_rows
    out_max_down = [None] * total_rows
    out_max_down_norm = [None] * total_rows

    level_1_count = 0
    level_2_count = 0
    noise_count = 0
    session_rows_count = 0

    for session_idx, (indices, _) in enumerate(sessions):
        if len(indices) == 0:
            continue

        session_rows_count += len(indices)

        for idx in indices:
            # Random level
            level = np.random.choice([1, 2, 3, 4])
            levels[idx] = level
            events[idx] = np.random.choice(["SPAWN", "EXTENSION", "REVERSAL"])

            # Target: out_max_down based on level
            if level == 2:
                level_2_count += 1
                out_max_down[idx] = -3.0  # Danger, deep crash
                out_max_down_norm[idx] = -3.0 / default_adr
                directions[idx] = "DOWN"
            elif level == 1:
                level_1_count += 1
                out_max_down[idx] = -0.1  # Safe
                out_max_down_norm[idx] = -0.1 / default_adr
                directions[idx] = np.random.choice(["UP", "DOWN"])
            else:
                noise_count += 1
                out_max_down[idx] = np.random.uniform(-1.5, -0.5)  # Noise
                out_max_down_norm[idx] = out_max_down[idx] / default_adr
                directions[idx] = np.random.choice(["UP", "DOWN"])

            # Uncorrelated noise for other outcomes
            out_next[idx] = np.random.uniform(-0.5, 0.5)
            out_next_norm[idx] = out_next[idx] / default_adr
            out_sess[idx] = np.random.uniform(-1.0, 1.0)
            out_sess_norm[idx] = out_sess[idx] / default_adr
            out_max_up[idx] = np.random.uniform(0.0, 2.0)
            out_max_up_norm[idx] = out_max_up[idx] / default_adr

    # Update columns
    result_df = result_df.with_columns([
        pl.Series(f"{prefix}_state_level", levels).cast(pl.Int32),
        pl.Series(f"{prefix}_state_event", events).cast(pl.Utf8),
        pl.Series(f"{prefix}_state_dir", directions).cast(pl.Utf8),
        pl.Series(f"{prefix}_out_next", out_next).cast(pl.Float64),
        pl.Series(f"{prefix}_out_next_norm", out_next_norm).cast(pl.Float64),
        pl.Series(f"{prefix}_out_sess", out_sess).cast(pl.Float64),
        pl.Series(f"{prefix}_out_sess_norm", out_sess_norm).cast(pl.Float64),
        pl.Series(f"{prefix}_out_max_up", out_max_up).cast(pl.Float64),
        pl.Series(f"{prefix}_out_max_up_norm", out_max_up_norm).cast(pl.Float64),
        pl.Series(f"{prefix}_out_max_down", out_max_down).cast(pl.Float64),
        pl.Series(f"{prefix}_out_max_down_norm", out_max_down_norm).cast(pl.Float64),
    ])

    # Save output
    output_path = output_dir / "lstm" / "bridged"
    output_path.mkdir(parents=True, exist_ok=True)

    parquet_file = output_path / f"test_max_down_{timeframe}.parquet"
    result_df.write_parquet(parquet_file, compression="zstd")

    print(f"  -> Saved {total_rows:,} rows to {parquet_file}")
    print(f"  -> Target: out_max_down (normalized by ADR={default_adr})")
    print(f"  -> Session rows: {session_rows_count:,} / {total_rows:,} ({100*session_rows_count/total_rows:.1f}%)")
    print(f"  -> Pattern: level=2 -> -3.0 (danger), level=1 -> -0.1 (safe)")
    print(f"  -> Distribution: {level_1_count:,} safe (L1), {level_2_count:,} danger (L2), {noise_count:,} noise (L3-4)")

    return parquet_file


# ============================================================================
# NEW: Target-Specific Validation Tests (Pure Synthetic)
# ============================================================================


def generate_test_next(
    output_dir: Path,
    n_sessions: int = 2000,
    timeframe: str = "M5",
    adr_value: float = 1.0,
    seed: int = 42,
) -> Path:
    """
    Generate Test: Next Bar Prediction (test_next.parquet).

    Tests if the model can predict the immediate next move (out_next).

    Rule:
        state_dir = UP (1)   -> out_next = +0.5 (ADR normalized)
        state_dir = DOWN (2) -> out_next = -0.5 (ADR normalized)

    Adversarial Trap:
        - out_sess is set to OPPOSITE direction (if next is +0.5, sess is -2.0)
        - out_max_up is set to misleading values
        - If model cheats by looking at other outcomes, it will fail

    Args:
        output_dir: Working directory for output
        n_sessions: Number of sessions to generate
        timeframe: Bar timeframe
        adr_value: ADR value (default 1.0 for simple math)
        seed: Random seed

    Returns:
        Path to generated parquet file
    """
    seq_len = get_bars_per_session(timeframe)
    print(f"Generating Test Next data: {n_sessions} sessions x {seq_len} bars ({timeframe})...")

    np.random.seed(seed)
    total_rows = n_sessions * seq_len

    # Create base schema with ADR
    df = create_base_schema(total_rows, adr_value, timeframe, seed)

    # Generate random directions (1=UP, 2=DOWN encoded as strings)
    direction_ids = np.random.choice([1, 2], size=total_rows)
    directions = np.where(direction_ids == 1, "UP", "DOWN")

    # Target outcome based on direction
    # UP -> +0.5, DOWN -> -0.5
    out_next = np.where(direction_ids == 1, 0.5, -0.5)
    out_next_norm = out_next / adr_value

    # ADVERSARIAL TRAP: Set out_sess to OPPOSITE direction
    # If next is +0.5 (UP), sess is -2.0 (DOWN crash)
    # If next is -0.5 (DOWN), sess is +2.0 (UP rally)
    out_sess = np.where(direction_ids == 1, -2.0, 2.0)
    out_sess_norm = out_sess / adr_value

    # More traps: misleading max values
    out_max_up = np.where(direction_ids == 1, -1.0, 3.0)  # Opposite of what you'd expect
    out_max_up_norm = out_max_up / adr_value
    out_max_down = np.where(direction_ids == 1, -3.0, 0.0)  # Also misleading
    out_max_down_norm = out_max_down / adr_value

    # Random levels and events (not used in this test's pattern, but needed for features)
    levels = np.random.choice([1, 2, 3, 4], size=total_rows)
    events = np.random.choice(["SPAWN", "EXTENSION", "REVERSAL"], size=total_rows)

    # Update ALL session columns with test pattern
    for prefix in SESSION_PREFIXES:
        df = df.with_columns([
            pl.Series(f"{prefix}_state_level", levels.astype(np.int32)),
            pl.Series(f"{prefix}_state_event", events),
            pl.Series(f"{prefix}_state_dir", directions),
            # Raw outcomes
            pl.Series(f"{prefix}_out_next", out_next),
            pl.Series(f"{prefix}_out_sess", out_sess),
            pl.Series(f"{prefix}_out_max_up", out_max_up),
            pl.Series(f"{prefix}_out_max_down", out_max_down),
            # Normalized outcomes
            pl.Series(f"{prefix}_out_next_norm", out_next_norm),
            pl.Series(f"{prefix}_out_sess_norm", out_sess_norm),
            pl.Series(f"{prefix}_out_max_up_norm", out_max_up_norm),
            pl.Series(f"{prefix}_out_max_down_norm", out_max_down_norm),
        ])

    # Save to parquet
    output_path = output_dir / "lstm" / "bridged"
    output_path.mkdir(parents=True, exist_ok=True)

    parquet_file = output_path / "synthetic_test_next.parquet"
    df.write_parquet(parquet_file, compression="zstd")

    up_count = np.sum(direction_ids == 1)
    down_count = total_rows - up_count

    print(f"  -> Saved {total_rows:,} rows to {parquet_file}")
    print(f"  -> Target: out_next (normalized by ADR={adr_value})")
    print(f"  -> Pattern: state_dir=UP -> +0.5, state_dir=DOWN -> -0.5")
    print(f"  -> Adversarial: out_sess set to OPPOSITE direction (trap)")
    print(f"  -> Distribution: {up_count:,} UP, {down_count:,} DOWN")

    return parquet_file


def generate_test_close(
    output_dir: Path,
    n_sessions: int = 2000,
    timeframe: str = "M5",
    adr_value: float = 1.0,
    seed: int = 42,
) -> Path:
    """
    Generate Test: Session Close Prediction (test_close.parquet).

    Tests if the model can predict the final session close (out_sess).
    The key is at Bar 0 only - forces model to use attention to look back.

    Rule:
        Bar 0 state_level = 1 -> out_sess = +2.0 (End of day is High)
        Bar 0 state_level = 2 -> out_sess = -2.0 (End of day is Low)
        Bars 1-N have noise levels (3, 4)

    Differentiation:
        - out_max_up is set to 0.0 (forces learning "where we end" not "how high")
        - out_next is random noise (not correlated with session close)

    Args:
        output_dir: Working directory for output
        n_sessions: Number of sessions to generate
        timeframe: Bar timeframe
        adr_value: ADR value (default 1.0 for simple math)
        seed: Random seed

    Returns:
        Path to generated parquet file
    """
    seq_len = get_bars_per_session(timeframe)
    print(f"Generating Test Close data: {n_sessions} sessions x {seq_len} bars ({timeframe})...")

    np.random.seed(seed)
    total_rows = n_sessions * seq_len

    # Create base schema with ADR
    df = create_base_schema(total_rows, adr_value, timeframe, seed)

    # Per-session "key" at Bar 0 (level 1 or 2)
    session_keys = np.random.choice([1, 2], size=n_sessions)
    session_outcomes = np.where(session_keys == 1, 2.0, -2.0)

    # Create level array: noise everywhere (levels 3-4)
    levels = np.random.choice([3, 4], size=total_rows)

    # Plant the key at Bar 0 of each session
    start_indices = np.arange(0, total_rows, seq_len)
    levels[start_indices] = session_keys

    # Expand session outcomes to all bars (same outcome for entire session)
    out_sess = np.repeat(session_outcomes, seq_len)
    out_sess_norm = out_sess / adr_value

    # Differentiation: out_max_up is always 0.0
    out_max_up = np.zeros(total_rows)
    out_max_up_norm = out_max_up / adr_value

    # Noise: out_next is random, uncorrelated with session close
    out_next = np.random.uniform(-0.5, 0.5, total_rows)
    out_next_norm = out_next / adr_value

    # out_max_down is also noise
    out_max_down = np.random.uniform(-1.0, 0.0, total_rows)
    out_max_down_norm = out_max_down / adr_value

    # Random events
    events = np.random.choice(["SPAWN", "EXTENSION", "REVERSAL"], size=total_rows)
    directions = np.where(out_sess > 0, "UP", "DOWN")

    # Update ALL session columns with test pattern
    for prefix in SESSION_PREFIXES:
        df = df.with_columns([
            pl.Series(f"{prefix}_state_level", levels.astype(np.int32)),
            pl.Series(f"{prefix}_state_event", events),
            pl.Series(f"{prefix}_state_dir", directions),
            # Raw outcomes
            pl.Series(f"{prefix}_out_next", out_next),
            pl.Series(f"{prefix}_out_sess", out_sess),
            pl.Series(f"{prefix}_out_max_up", out_max_up),
            pl.Series(f"{prefix}_out_max_down", out_max_down),
            # Normalized outcomes
            pl.Series(f"{prefix}_out_next_norm", out_next_norm),
            pl.Series(f"{prefix}_out_sess_norm", out_sess_norm),
            pl.Series(f"{prefix}_out_max_up_norm", out_max_up_norm),
            pl.Series(f"{prefix}_out_max_down_norm", out_max_down_norm),
        ])

    # Save to parquet
    output_path = output_dir / "lstm" / "bridged"
    output_path.mkdir(parents=True, exist_ok=True)

    parquet_file = output_path / "synthetic_test_close.parquet"
    df.write_parquet(parquet_file, compression="zstd")

    key_1_count = np.sum(session_keys == 1)
    key_2_count = n_sessions - key_1_count

    print(f"  -> Saved {total_rows:,} rows to {parquet_file}")
    print(f"  -> Target: out_sess (normalized by ADR={adr_value})")
    print(f"  -> Pattern: bar[0].level=1 -> +2.0, bar[0].level=2 -> -2.0")
    print(f"  -> Bars 1-{seq_len-1} contain noise (levels 3-4)")
    print(f"  -> Differentiation: out_max_up = 0.0 (forces 'where we end' learning)")
    print(f"  -> Distribution: {key_1_count} key=1 sessions, {key_2_count} key=2 sessions")

    return parquet_file


def generate_test_max_up(
    output_dir: Path,
    n_sessions: int = 3000,
    timeframe: str = "M5",
    adr_value: float = 1.0,
    seed: int = 42,
) -> Path:
    """
    Generate Test: Max Upside Prediction (test_max_up.parquet).

    Tests if the model can predict the highest point reached (out_max_up).
    Specifically tests Level 1 + SPAWN combination.

    Rule:
        state_level = 1 AND state_event = "SPAWN":
            -> out_max_up = +5.0 (Huge wick to the upside)
            -> out_sess = 0.0 (Price crashes back to start - flat close)
        All other combinations:
            -> out_max_up = +0.5 (Small move)
            -> out_sess = random (uncorrelated)

    Why: Proves model can spot a temporary spike even if the day ends flat.

    Args:
        output_dir: Working directory for output
        n_sessions: Number of sessions to generate
        timeframe: Bar timeframe
        adr_value: ADR value (default 1.0 for simple math)
        seed: Random seed

    Returns:
        Path to generated parquet file
    """
    seq_len = get_bars_per_session(timeframe)
    print(f"Generating Test Max Up data: {n_sessions} sessions x {seq_len} bars ({timeframe})...")

    np.random.seed(seed)
    total_rows = n_sessions * seq_len

    # Create base schema with ADR
    df = create_base_schema(total_rows, adr_value, timeframe, seed)

    # Generate random levels (1-4) and events
    levels = np.random.choice([1, 2, 3, 4], size=total_rows)
    events = np.random.choice(["SPAWN", "EXTENSION", "REVERSAL"], size=total_rows)

    # Logic rule: Level=1 AND Event="SPAWN" -> big spike
    is_spike = (levels == 1) & (events == "SPAWN")

    # Target: out_max_up
    out_max_up = np.where(is_spike, 5.0, 0.5)
    out_max_up_norm = out_max_up / adr_value

    # Key differentiation: out_sess is 0.0 for spikes (price crashes back)
    # For non-spikes, out_sess is random
    out_sess = np.where(is_spike, 0.0, np.random.uniform(-1.0, 1.0, total_rows))
    out_sess_norm = out_sess / adr_value

    # out_next is uncorrelated noise
    out_next = np.random.uniform(-0.5, 0.5, total_rows)
    out_next_norm = out_next / adr_value

    # out_max_down is also uncorrelated
    out_max_down = np.random.uniform(-2.0, 0.0, total_rows)
    out_max_down_norm = out_max_down / adr_value

    directions = np.where(is_spike, "UP", np.random.choice(["UP", "DOWN"], size=total_rows))

    # Update ALL session columns with test pattern
    for prefix in SESSION_PREFIXES:
        df = df.with_columns([
            pl.Series(f"{prefix}_state_level", levels.astype(np.int32)),
            pl.Series(f"{prefix}_state_event", events),
            pl.Series(f"{prefix}_state_dir", directions),
            # Raw outcomes
            pl.Series(f"{prefix}_out_next", out_next),
            pl.Series(f"{prefix}_out_sess", out_sess),
            pl.Series(f"{prefix}_out_max_up", out_max_up),
            pl.Series(f"{prefix}_out_max_down", out_max_down),
            # Normalized outcomes
            pl.Series(f"{prefix}_out_next_norm", out_next_norm),
            pl.Series(f"{prefix}_out_sess_norm", out_sess_norm),
            pl.Series(f"{prefix}_out_max_up_norm", out_max_up_norm),
            pl.Series(f"{prefix}_out_max_down_norm", out_max_down_norm),
        ])

    # Save to parquet
    output_path = output_dir / "lstm" / "bridged"
    output_path.mkdir(parents=True, exist_ok=True)

    parquet_file = output_path / "synthetic_test_max_up.parquet"
    df.write_parquet(parquet_file, compression="zstd")

    spike_count = np.sum(is_spike)
    normal_count = total_rows - spike_count
    spike_pct = 100.0 * spike_count / total_rows

    print(f"  -> Saved {total_rows:,} rows to {parquet_file}")
    print(f"  -> Target: out_max_up (normalized by ADR={adr_value})")
    print(f"  -> Pattern: Level=1 AND Event=SPAWN -> +5.0, else +0.5")
    print(f"  -> Differentiation: Spikes have out_sess=0.0 (flat close despite spike)")
    print(f"  -> Distribution: {spike_count:,} spikes ({spike_pct:.1f}%), {normal_count:,} normal")

    return parquet_file


def generate_test_max_down(
    output_dir: Path,
    n_sessions: int = 2000,
    timeframe: str = "M5",
    adr_value: float = 1.0,
    seed: int = 42,
) -> Path:
    """
    Generate Test: Max Drawdown Prediction (test_max_down.parquet).

    Tests if the model can predict drawdown risk (out_max_down).
    Level 2 indicates danger.

    Rule:
        state_level = 2 -> out_max_down = -3.0 (Deep crash)
        state_level = 1 -> out_max_down = -0.1 (Safe, minimal drawdown)
        state_level = 3,4 -> random between -0.5 and -1.5 (noise)

    Why: Ensures model specifically learns to predict *how low* price will go.

    Args:
        output_dir: Working directory for output
        n_sessions: Number of sessions to generate
        timeframe: Bar timeframe
        adr_value: ADR value (default 1.0 for simple math)
        seed: Random seed

    Returns:
        Path to generated parquet file
    """
    seq_len = get_bars_per_session(timeframe)
    print(f"Generating Test Max Down data: {n_sessions} sessions x {seq_len} bars ({timeframe})...")

    np.random.seed(seed)
    total_rows = n_sessions * seq_len

    # Create base schema with ADR
    df = create_base_schema(total_rows, adr_value, timeframe, seed)

    # Generate random levels
    levels = np.random.choice([1, 2, 3, 4], size=total_rows)

    # Target: out_max_down based on level
    out_max_down = np.where(
        levels == 2, -3.0,  # Level 2 = danger, deep crash
        np.where(
            levels == 1, -0.1,  # Level 1 = safe
            np.random.uniform(-1.5, -0.5, total_rows)  # Levels 3,4 = noise
        )
    )
    out_max_down_norm = out_max_down / adr_value

    # Other outcomes are noise (uncorrelated with max_down)
    out_next = np.random.uniform(-0.5, 0.5, total_rows)
    out_next_norm = out_next / adr_value

    out_sess = np.random.uniform(-1.0, 1.0, total_rows)
    out_sess_norm = out_sess / adr_value

    out_max_up = np.random.uniform(0.0, 2.0, total_rows)
    out_max_up_norm = out_max_up / adr_value

    # Random events
    events = np.random.choice(["SPAWN", "EXTENSION", "REVERSAL"], size=total_rows)
    directions = np.where(levels == 2, "DOWN", np.random.choice(["UP", "DOWN"], size=total_rows))

    # Update ALL session columns with test pattern
    for prefix in SESSION_PREFIXES:
        df = df.with_columns([
            pl.Series(f"{prefix}_state_level", levels.astype(np.int32)),
            pl.Series(f"{prefix}_state_event", events),
            pl.Series(f"{prefix}_state_dir", directions),
            # Raw outcomes
            pl.Series(f"{prefix}_out_next", out_next),
            pl.Series(f"{prefix}_out_sess", out_sess),
            pl.Series(f"{prefix}_out_max_up", out_max_up),
            pl.Series(f"{prefix}_out_max_down", out_max_down),
            # Normalized outcomes
            pl.Series(f"{prefix}_out_next_norm", out_next_norm),
            pl.Series(f"{prefix}_out_sess_norm", out_sess_norm),
            pl.Series(f"{prefix}_out_max_up_norm", out_max_up_norm),
            pl.Series(f"{prefix}_out_max_down_norm", out_max_down_norm),
        ])

    # Save to parquet
    output_path = output_dir / "lstm" / "bridged"
    output_path.mkdir(parents=True, exist_ok=True)

    parquet_file = output_path / "synthetic_test_max_down.parquet"
    df.write_parquet(parquet_file, compression="zstd")

    level_1_count = np.sum(levels == 1)
    level_2_count = np.sum(levels == 2)
    noise_count = total_rows - level_1_count - level_2_count

    print(f"  -> Saved {total_rows:,} rows to {parquet_file}")
    print(f"  -> Target: out_max_down (normalized by ADR={adr_value})")
    print(f"  -> Pattern: level=2 -> -3.0 (danger), level=1 -> -0.1 (safe)")
    print(f"  -> Distribution: {level_1_count:,} safe (L1), {level_2_count:,} danger (L2), {noise_count:,} noise (L3-4)")

    return parquet_file


def run_validation(
    working_dir: Path,
    test_name: str = "sanity",
    num_epochs: int = 30,
    target_session: str = "lon",
    sequence_length: int = 64,
) -> dict:
    """
    Run validation: train on synthetic data and check if pattern is learned.

    Args:
        working_dir: Working directory containing synthetic parquets
        test_name: "sanity", "memory", "logic", "next", "close", "max_up", "max_down"
        num_epochs: Number of epochs to train
        target_session: Session prefix
        sequence_length: Sequence length (should match generation)

    Returns:
        Dict with test results
    """
    from app.transformer.config import TransformerConfig
    from app.transformer.trainer import TransformerTrainer

    # Find the synthetic parquet
    parquet_path = working_dir / "lstm" / "bridged" / f"synthetic_test_{test_name}.parquet"

    if not parquet_path.exists():
        print(f"Error: Synthetic parquet not found at {parquet_path}")
        print("Run 'generate' first to create test data.")
        return {"test": test_name, "passed": False, "error": "Parquet not found"}

    print(f"\n{'='*60}")
    print(f"Running Validation: {test_name.upper()}")
    print(f"{'='*60}")

    # Map test_name to target_outcome for the dataset
    # Original tests use max_up, new tests use their specific target
    test_to_outcome = {
        "sanity": "max_up",
        "memory": "max_up",
        "logic": "max_up",
        "next": "next",
        "close": "sess",
        "max_up": "max_up",
        "max_down": "max_down",
    }
    target_outcome = test_to_outcome.get(test_name, "max_up")

    # Configure for quick validation
    config = TransformerConfig(
        target_session=target_session,
        target_outcome=target_outcome,
        sequence_length=sequence_length,
        batch_size=64,
        d_model=64,       # Smaller model for fast validation
        n_layers=2,       # Minimal layers
        n_heads=2,
        num_epochs=num_epochs,
        learning_rate=1e-3,  # Higher LR for faster convergence
        early_stopping_patience=10,
        save_every=100,   # Don't save checkpoints
    )

    trainer = TransformerTrainer(
        config=config,
        working_directory=working_dir,
        model_name=f"validation_{test_name}",
    )

    # Setup with specific parquet file
    # Use same file for train/val since we only have 1 file (ok for validation purposes)
    trainer.setup(parquet_files=[parquet_path, parquet_path], val_ratio=0.5)

    # Train
    state = trainer.train()

    # Determine pass/fail thresholds
    # New target-specific tests have their own thresholds
    thresholds = {
        # Original tests
        "sanity": 0.001,  # Sanity should achieve near-zero loss
        "memory": 0.01,   # Memory is harder (needs attention)
        "logic": 0.005,   # Logic requires feature combination
        # New target-specific tests
        "next": 0.01,     # Simple direction -> next mapping
        "close": 0.02,    # Requires attention to Bar 0
        "max_up": 0.05,   # Level + Event combination (8.3% spikes)
        "max_down": 0.02, # Level -> max_down mapping
    }
    threshold = thresholds.get(test_name, 0.01)

    passed = state.best_loss < threshold

    # Results
    result = {
        "test": test_name,
        "passed": passed,
        "best_loss": state.best_loss,
        "threshold": threshold,
        "final_epoch": state.epoch,
        "epochs_trained": len(state.train_losses),
        "target_outcome": target_outcome,
    }

    print(f"\n{'='*60}")
    print(f"RESULT: {'PASSED' if passed else 'FAILED'}")
    print(f"Best Loss: {state.best_loss:.6f} (threshold: {threshold})")
    print(f"Target Outcome: {target_outcome}")
    print(f"{'='*60}")

    return result


def run_all_validations(working_dir: Path, include_target_tests: bool = True) -> list[dict]:
    """
    Run all validation tests and return results.

    Args:
        working_dir: Working directory
        include_target_tests: If True, also run the 4 new target-specific tests
    """
    results = []

    # Generate test data
    print("\n" + "="*60)
    print("GENERATING SYNTHETIC TEST DATA")
    print("="*60 + "\n")

    # Original tests
    generate_sanity_check(working_dir)
    generate_memory_test(working_dir)
    generate_logic_test(working_dir)

    # New target-specific tests
    if include_target_tests:
        generate_test_next(working_dir)
        generate_test_close(working_dir)
        generate_test_max_up(working_dir)
        generate_test_max_down(working_dir)

    # Run tests
    print("\n" + "="*60)
    print("RUNNING VALIDATION TESTS")
    print("="*60)

    # Original tests
    results.append(run_validation(working_dir, "sanity", num_epochs=20))
    results.append(run_validation(working_dir, "memory", num_epochs=50))
    results.append(run_validation(working_dir, "logic", num_epochs=30))

    # New target-specific tests
    if include_target_tests:
        results.append(run_validation(working_dir, "next", num_epochs=30))
        results.append(run_validation(working_dir, "close", num_epochs=50))
        results.append(run_validation(working_dir, "max_up", num_epochs=40))
        results.append(run_validation(working_dir, "max_down", num_epochs=30))

    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    all_passed = True
    for r in results:
        status = "PASSED" if r["passed"] else "FAILED"
        target = r.get("target_outcome", "max_up")
        print(f"  {r['test']:10s}: {status} (loss={r['best_loss']:.6f}, target={target})")
        if not r["passed"]:
            all_passed = False

    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Transformer model validation via synthetic data"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # All available test choices
    ALL_TESTS = ["sanity", "memory", "logic", "next", "close", "max_up", "max_down", "all"]

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate synthetic test data")
    gen_parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        required=True,
        help="Working directory for output"
    )
    gen_parser.add_argument(
        "--test", "-t",
        choices=ALL_TESTS,
        default="all",
        help="Which test data to generate (next/close/max_up/max_down are target-specific tests)"
    )
    gen_parser.add_argument("--sessions", type=int, help="Number of sessions (default varies by test)")
    gen_parser.add_argument("--timeframe", type=str, default="M5", help="Timeframe (M1, M5, M10, etc.)")
    gen_parser.add_argument("--adr", type=float, default=1.0, help="ADR value (default 1.0 for simple math)")
    gen_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run validation (generate + train + test)")
    run_parser.add_argument(
        "--working-dir", "-w",
        type=Path,
        required=True,
        help="Working directory"
    )
    run_parser.add_argument(
        "--test", "-t",
        choices=ALL_TESTS,
        default="all",
        help="Which test to run (next/close/max_up/max_down are target-specific tests)"
    )
    run_parser.add_argument("--epochs", type=int, default=30, help="Training epochs")

    args = parser.parse_args()

    # Default session counts for each test
    DEFAULT_TEST_SESSIONS = {
        "sanity": 1000,
        "memory": 2000,
        "logic": 5000,
        "next": 2000,
        "close": 2000,
        "max_up": 3000,
        "max_down": 2000,
    }

    # Generator mapping
    GENERATORS = {
        "sanity": generate_sanity_check,
        "memory": generate_memory_test,
        "logic": generate_logic_test,
        "next": generate_test_next,
        "close": generate_test_close,
        "max_up": generate_test_max_up,
        "max_down": generate_test_max_down,
    }

    if args.command == "generate":
        output_dir = args.output_dir
        timeframe = args.timeframe
        adr = args.adr
        seed = args.seed

        if args.test == "all":
            # Generate all tests
            for test_name, gen_func in GENERATORS.items():
                n = args.sessions if args.sessions else DEFAULT_TEST_SESSIONS[test_name]
                gen_func(output_dir, n, timeframe, adr, seed)
        else:
            # Generate specific test
            n = args.sessions if args.sessions else DEFAULT_TEST_SESSIONS[args.test]
            GENERATORS[args.test](output_dir, n, timeframe, adr, seed)

        print("\nDone! Synthetic test data generated.")

    elif args.command == "run":
        working_dir = args.working_dir

        if args.test == "all":
            run_all_validations(working_dir)
        else:
            # Generate specific test data first
            n = DEFAULT_TEST_SESSIONS[args.test]
            GENERATORS[args.test](working_dir, n)

            # Run the test
            run_validation(working_dir, args.test, args.epochs)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
