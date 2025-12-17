"""
Test script for Edge Finder Phase 0 validation.

Run from backend directory:
    python test_edge_finder.py
"""

import sys
sys.path.insert(0, '.')

from app.edge_finder.generator import generate_test_dataset, get_available_pairs
from app.edge_finder.storage import (
    get_session_stats,
    list_session_files,
    load_session_dataset,
)
import numpy as np


def main():
    print("=" * 60)
    print("Edge Finder Phase 0 - Test Dataset Generation")
    print("=" * 60)

    # Check available pairs
    pairs = get_available_pairs()
    print(f"\nAvailable pairs: {pairs}")

    if not pairs:
        print("ERROR: No cached data found. Run the pipeline first.")
        return

    # Use first available pair or EURUSD if available
    test_pair = "EURUSD" if "EURUSD" in pairs else pairs[0]
    print(f"\nUsing pair: {test_pair}")

    # Generate test dataset (100 sessions)
    print("\nGenerating test dataset (100 sessions, NY session, M10)...")
    stats = generate_test_dataset(
        pair=test_pair,
        timeframe="M10",
        session_type="ny",
        max_sessions=100
    )

    print("\n" + "-" * 40)
    print("Generation Statistics:")
    print("-" * 40)
    print(f"  Processed: {stats['processed']}")
    print(f"  Saved:     {stats['saved']}")
    print(f"  Skipped:   {stats['skipped']}")
    print(f"  Errors:    {stats['errors']}")

    if stats['error_details']:
        print("\nFirst 3 errors:")
        for err in stats['error_details'][:3]:
            print(f"  Date: {err['date']}, Error: {err['error']}")

    # Show overall session stats
    session_stats = get_session_stats()
    print("\n" + "-" * 40)
    print("Overall Session Statistics:")
    print("-" * 40)
    print(f"  Total sessions: {session_stats['total_sessions']}")
    print(f"  By pair: {session_stats['by_pair']}")
    print(f"  By session type: {session_stats['by_session_type']}")
    print(f"  By timeframe: {session_stats['by_timeframe']}")

    # Validate a sample session
    session_ids = list_session_files()
    if session_ids:
        print("\n" + "-" * 40)
        print("Sample Session Validation:")
        print("-" * 40)

        sample_id = session_ids[0]
        print(f"  Loading: {sample_id}")

        dataset = load_session_dataset(sample_id)
        if dataset:
            print(f"  Session ID: {dataset.session_id}")
            print(f"  Pair: {dataset.pair}")
            print(f"  Date: {dataset.session_date}")
            print(f"  Session Type: {dataset.session_type}")
            print(f"  Timeframe: {dataset.timeframe}")
            print(f"  Total Bars: {dataset.total_bars}")
            print(f"  Matrix Shape: {dataset.matrix.shape}")
            print(f"  Session Start ATR: {dataset.session_start_atr:.6f}")
            print(f"  Metadata Count: {len(dataset.metadata)}")

            # Validate matrix values
            print("\n  Matrix Statistics:")
            for i, name in enumerate(["L1_Dir", "L1_Amp", "L1_Dur", "L1_Leg",
                                       "L2_Dir", "L2_Amp", "L2_Dur", "L2_Leg",
                                       "L3_Dir", "L3_Amp", "L3_Dur", "L3_Leg",
                                       "L4_Dir", "L4_Amp", "L4_Dur", "L4_Leg",
                                       "L5_Dir", "L5_Amp", "L5_Dur", "L5_Leg"]):
                col = dataset.matrix[:, i]
                non_zero = np.count_nonzero(col)
                if non_zero > 0:
                    print(f"    {name}: min={col.min():.4f}, max={col.max():.4f}, non_zero={non_zero}/{len(col)}")

            # Sample metadata
            print("\n  Sample Metadata (first 3 bars):")
            for meta in dataset.metadata[:3]:
                print(f"    Bar {meta.bar_index}: next_move={meta.next_bar_move_atr:.4f}, "
                      f"drift={meta.session_drift_atr:.4f}, "
                      f"mae={meta.mae_to_session_end:.4f}")
        else:
            print("  ERROR: Failed to load session")

    print("\n" + "=" * 60)
    print("Phase 0 Validation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
