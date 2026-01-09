"""
Synthetic data generation and validation for Transformer model.

Creates test parquets with known input→output patterns to verify:
1. Sanity Check: Model can map single input to output
2. Memory Test: Model attention can look back to sequence start
3. Logic Test: Model can combine multiple features

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
    adr_value: float = 0.0050,
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
            # Outcome columns
            pl.lit(0.0).alias(f"{prefix}_out_next"),
            pl.lit(0.0).alias(f"{prefix}_out_sess"),
            pl.lit(0.0).alias(f"{prefix}_out_max_up"),
            pl.lit(0.0).alias(f"{prefix}_out_max_down"),
        ])

    return df


def generate_sanity_check(
    output_dir: Path,
    n_sessions: int = 1000,
    timeframe: str = "M5",
    adr_value: float = 0.0050,
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
        adr_value: ADR value for all sessions
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

    # Update ALL session columns with test pattern
    for prefix in SESSION_PREFIXES:
        df = df.with_columns([
            pl.Series(f"{prefix}_state_level", levels.astype(np.int32)),
            pl.Series(f"{prefix}_state_event", events),
            pl.Series(f"{prefix}_state_dir", directions),
            pl.Series(f"{prefix}_out_max_up", outcomes),
        ])

    # Save to parquet
    output_path = output_dir / "lstm" / "bridged"
    output_path.mkdir(parents=True, exist_ok=True)

    parquet_file = output_path / "synthetic_test_sanity.parquet"
    df.write_parquet(parquet_file, compression="zstd")

    print(f"  -> Saved {total_rows:,} rows to {parquet_file}")
    print(f"  -> Pattern: level=1 -> +0.01 (UP), level=2 -> -0.01 (DOWN)")
    print(f"  -> All sessions populated: {', '.join(SESSION_PREFIXES)}")

    return parquet_file


def generate_memory_test(
    output_dir: Path,
    n_sessions: int = 2000,
    timeframe: str = "M5",
    adr_value: float = 0.0050,
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
        adr_value: ADR value for all sessions
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

    # Update ALL session columns with test pattern
    for prefix in SESSION_PREFIXES:
        df = df.with_columns([
            pl.Series(f"{prefix}_state_level", levels.astype(np.int32)),
            pl.Series(f"{prefix}_state_event", events),
            pl.Series(f"{prefix}_state_dir", directions),
            pl.Series(f"{prefix}_out_max_up", outcomes),
        ])

    # Save to parquet
    output_path = output_dir / "lstm" / "bridged"
    output_path.mkdir(parents=True, exist_ok=True)

    parquet_file = output_path / "synthetic_test_memory.parquet"
    df.write_parquet(parquet_file, compression="zstd")

    print(f"  -> Saved {total_rows:,} rows to {parquet_file}")
    print(f"  -> Pattern: bar[0].level=1 -> +0.02, bar[0].level=2 -> -0.02")
    print(f"  -> Bars 1-{seq_len-1} contain noise (levels 3-4)")
    print(f"  -> All sessions populated: {', '.join(SESSION_PREFIXES)}")

    return parquet_file


def generate_logic_test(
    output_dir: Path,
    n_sessions: int = 5000,
    timeframe: str = "M5",
    adr_value: float = 0.0050,
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
        adr_value: ADR value for all sessions
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

    # Update ALL session columns with test pattern
    for prefix in SESSION_PREFIXES:
        df = df.with_columns([
            pl.Series(f"{prefix}_state_level", levels.astype(np.int32)),
            pl.Series(f"{prefix}_state_event", events),
            pl.Series(f"{prefix}_state_dir", directions),
            pl.Series(f"{prefix}_out_max_up", outcomes),
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
    print(f"  -> Pattern: Level=1 AND Event=SPAWN -> +0.01 (UP), else -0.01 (DOWN)")
    print(f"  -> Distribution: {up_count:,} UP ({up_pct:.1f}%), {down_count:,} DOWN ({100-up_pct:.1f}%)")
    print(f"  -> All sessions populated: {', '.join(SESSION_PREFIXES)}")

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
        test_name: "sanity", "memory", or "logic"
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

    # Configure for quick validation
    config = TransformerConfig(
        target_session=target_session,
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
    thresholds = {
        "sanity": 0.001,  # Sanity should achieve near-zero loss
        "memory": 0.01,   # Memory is harder (needs attention)
        "logic": 0.005,   # Logic requires feature combination
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
    }

    print(f"\n{'='*60}")
    print(f"RESULT: {'PASSED' if passed else 'FAILED'}")
    print(f"Best Loss: {state.best_loss:.6f} (threshold: {threshold})")
    print(f"{'='*60}")

    return result


def run_all_validations(working_dir: Path) -> list[dict]:
    """Run all validation tests and return results."""
    results = []

    # Generate test data
    print("\n" + "="*60)
    print("GENERATING SYNTHETIC TEST DATA")
    print("="*60 + "\n")

    generate_sanity_check(working_dir)
    generate_memory_test(working_dir)
    generate_logic_test(working_dir)

    # Run tests
    print("\n" + "="*60)
    print("RUNNING VALIDATION TESTS")
    print("="*60)

    results.append(run_validation(working_dir, "sanity", num_epochs=20))
    results.append(run_validation(working_dir, "memory", num_epochs=50))
    results.append(run_validation(working_dir, "logic", num_epochs=30))

    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    all_passed = True
    for r in results:
        status = "PASSED" if r["passed"] else "FAILED"
        print(f"  {r['test']:10s}: {status} (loss={r['best_loss']:.6f})")
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
        choices=["sanity", "memory", "logic", "all"],
        default="all",
        help="Which test data to generate"
    )
    gen_parser.add_argument("--sessions", type=int, help="Number of sessions (default varies by test)")
    gen_parser.add_argument("--timeframe", type=str, default="M5", help="Timeframe (M1, M5, M10, etc.)")
    gen_parser.add_argument("--adr", type=float, default=0.0050, help="ADR value")
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
        choices=["sanity", "memory", "logic", "all"],
        default="all",
        help="Which test to run"
    )
    run_parser.add_argument("--epochs", type=int, default=30, help="Training epochs")

    args = parser.parse_args()

    if args.command == "generate":
        output_dir = args.output_dir
        timeframe = args.timeframe
        adr = args.adr
        seed = args.seed

        if args.test == "all" or args.test == "sanity":
            n = args.sessions if args.sessions else DEFAULT_SESSIONS["sanity"]
            generate_sanity_check(output_dir, n, timeframe, adr, seed)
        if args.test == "all" or args.test == "memory":
            n = args.sessions if args.sessions else DEFAULT_SESSIONS["memory"]
            generate_memory_test(output_dir, n, timeframe, adr, seed)
        if args.test == "all" or args.test == "logic":
            n = args.sessions if args.sessions else DEFAULT_SESSIONS["logic"]
            generate_logic_test(output_dir, n, timeframe, adr, seed)

        print("\nDone! Synthetic test data generated.")

    elif args.command == "run":
        working_dir = args.working_dir

        if args.test == "all":
            run_all_validations(working_dir)
        else:
            # Generate specific test data first
            if args.test == "sanity":
                generate_sanity_check(working_dir)
            elif args.test == "memory":
                generate_memory_test(working_dir)
            else:
                generate_logic_test(working_dir)

            # Run the test
            run_validation(working_dir, args.test, args.epochs)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
