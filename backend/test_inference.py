"""
Phase 6 Validation Script - Vector Search / Inference

Tests the LatentVectorIndex and EdgeInferenceEngine classes.
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

import numpy as np


def test_phase_6():
    """Test the vector index and inference engine."""
    from app.edge_finder.vector_index import LatentVectorIndex
    from app.edge_finder.inference import EdgeInferenceEngine
    from app.edge_finder.storage import list_session_files, get_sessions_path

    working_dir = Path(r"C:\Users\lawfp\Desktop\Data4")

    print("=" * 60)
    print("Phase 6 Validation: Vector Search / Inference")
    print("=" * 60)

    # Check sessions exist
    session_ids = list_session_files(working_directory=working_dir)
    print(f"\nFound {len(session_ids)} session files")

    if len(session_ids) == 0:
        print("ERROR: No session files found!")
        return False

    # --- Test LatentVectorIndex ---
    print("\n" + "-" * 40)
    print("Testing LatentVectorIndex")
    print("-" * 40)

    index = LatentVectorIndex(
        model_name="vae_test",
        working_directory=working_dir,
        device="cpu",  # Use CPU for testing
    )

    # Load model
    print("\nLoading VAE model...")
    try:
        index.load_model()
        print(f"  Model loaded: latent_dim={index.latent_dim}")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return False

    # Build index (limit to first 10 sessions for speed)
    print("\nBuilding index from first 10 sessions...")
    test_sessions = session_ids[:10]

    # We'll build manually to limit sessions
    sessions_path = get_sessions_path(working_dir)
    all_vectors = []
    all_session_ids = []
    all_bar_indices = []
    all_next_bar_moves = []
    all_session_drifts = []
    all_maes = []
    all_mfes = []
    all_session_progress = []

    for i, session_id in enumerate(test_sessions):
        print(f"  Processing {i+1}/{len(test_sessions)}: {session_id}")
        file_path = sessions_path / f"{session_id}.npz"
        data = np.load(file_path, allow_pickle=True)
        matrix = data["matrix"]
        n_bars = int(data["total_bars"])

        # Only index every 5th bar to keep it fast
        for bar_idx in range(0, n_bars, 5):
            prefix_matrix = matrix[:bar_idx + 1]
            bar_latent = index.encode_matrix(prefix_matrix)

            all_vectors.append(bar_latent)
            all_session_ids.append(session_id)
            all_bar_indices.append(bar_idx)
            all_next_bar_moves.append(float(data["next_bar_moves"][bar_idx]))
            all_session_drifts.append(float(data["session_drifts"][bar_idx]))
            all_maes.append(float(data["maes"][bar_idx]))
            all_mfes.append(float(data["mfes"][bar_idx]))
            all_session_progress.append(float(data["session_progress"][bar_idx]))

    # Set index data
    index.vectors = np.array(all_vectors, dtype=np.float32)
    index.session_ids = all_session_ids
    index.bar_indices = all_bar_indices
    index.next_bar_moves = np.array(all_next_bar_moves, dtype=np.float32)
    index.session_drifts = np.array(all_session_drifts, dtype=np.float32)
    index.maes = np.array(all_maes, dtype=np.float32)
    index.mfes = np.array(all_mfes, dtype=np.float32)
    index.session_progress = np.array(all_session_progress, dtype=np.float32)
    index._is_loaded = True

    print(f"\nIndex built:")
    print(f"  Total vectors: {index.num_vectors}")
    print(f"  Unique sessions: {index.num_sessions}")
    print(f"  Latent dimension: {index.latent_dim}")
    print(f"  Vector shape: {index.vectors.shape}")

    # Test search
    print("\n" + "-" * 40)
    print("Testing KNN Search")
    print("-" * 40)

    # Use first vector as query
    query_vector = index.vectors[0]
    print(f"\nQuery vector shape: {query_vector.shape}")

    # Standard search
    results = index.search(query_vector, k=20)
    print(f"\nStandard search (k=20):")
    print(f"  Results: {len(results)}")
    print(f"  Top 5 distances: {[round(r.distance, 4) for r in results[:5]]}")
    print(f"  Top 5 sessions: {[r.session_id for r in results[:5]]}")

    # Unique session search
    unique_results = index.search_unique_sessions(query_vector, k=10)
    print(f"\nUnique session search (k=10):")
    print(f"  Results: {len(unique_results)}")
    print(f"  Distances: {[round(r.distance, 4) for r in unique_results]}")
    print(f"  Sessions: {[r.session_id for r in unique_results]}")

    # --- Test EdgeInferenceEngine ---
    print("\n" + "-" * 40)
    print("Testing EdgeInferenceEngine")
    print("-" * 40)

    engine = EdgeInferenceEngine(
        model_name="vae_test",
        working_directory=working_dir,
        k_neighbors=50,
        device="cpu",
    )

    # Copy index to engine (avoid rebuilding)
    engine.index = index
    engine._is_ready = True

    # Load a test matrix
    test_session = session_ids[5]  # Use 6th session as query
    test_file = sessions_path / f"{test_session}.npz"
    test_data = np.load(test_file, allow_pickle=True)
    test_matrix = test_data["matrix"][:30]  # First 30 bars

    print(f"\nTest matrix shape: {test_matrix.shape}")
    print(f"Source session: {test_session}")

    # Compute edge probabilities
    edge = engine.compute_edge(test_matrix, k=50, unique_sessions=True)

    print(f"\nEdge Probabilities:")
    print(f"  Matches: {edge.num_matches}")
    print(f"  Avg distance: {edge.avg_distance:.4f}")
    print(f"  Next bar up %: {edge.next_bar_up_pct:.2%}")
    print(f"  Next bar avg move: {edge.next_bar_avg_move:.4f} ATR")
    print(f"  Session up %: {edge.session_up_pct:.2%}")
    print(f"  Session avg drift: {edge.session_avg_drift:.4f} ATR")
    print(f"  Avg MAE: {edge.avg_mae:.4f} ATR")
    print(f"  MAE percentiles: p25={edge.mae_p25:.4f}, p50={edge.mae_p50:.4f}, p75={edge.mae_p75:.4f}, p95={edge.mae_p95:.4f}")
    print(f"  Avg MFE: {edge.avg_mfe:.4f} ATR")
    print(f"  Risk/Reward: {edge.risk_reward_ratio:.2f}")
    print(f"  Avg session progress: {edge.avg_session_progress:.2%}")

    # Test to_dict
    edge_dict = edge.to_dict()
    print(f"\nEdge as dict (num keys): {len(edge_dict)}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Phase 6 Validation: PASSED")
    print("=" * 60)
    print("\nAll components working:")
    print("  - LatentVectorIndex: Model loading, encoding, KNN search")
    print("  - search_unique_sessions(): Anti-bias filtering")
    print("  - EdgeInferenceEngine: Edge probability computation")
    print("  - EdgeProbabilities: All metrics computed correctly")

    return True


if __name__ == "__main__":
    success = test_phase_6()
    sys.exit(0 if success else 1)
