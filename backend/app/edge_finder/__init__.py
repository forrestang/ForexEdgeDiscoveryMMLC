"""
Edge Finder package for pattern recognition and probability estimation.

This package transforms waveform visualizations into a data generation engine
for training VAE models and performing semantic pattern search.
"""

from app.edge_finder.matrix_serializer import (
    snapshot_to_row,
    snapshots_to_matrix,
    compute_session_start_atr,
    get_session_bar_count,
)

from app.edge_finder.future_truth import (
    BarMetadata,
    SessionDataset,
    generate_session_id,
    compute_bar_metadata,
    compute_session_dataset,
)

from app.edge_finder.storage import (
    get_edge_finder_path,
    get_sessions_path,
    save_session_dataset,
    load_session_dataset,
    list_session_files,
    get_session_stats,
)

from app.edge_finder.generator import (
    generate_session_dataset,
    generate_all_sessions,
    generate_test_dataset,
    iter_sessions,
    get_available_pairs,
)

from app.edge_finder.vector_index import (
    LatentVectorIndex,
    IndexedVector,
    SearchResult,
)

from app.edge_finder.inference import (
    EdgeInferenceEngine,
    EdgeProbabilities,
)

from app.edge_finder.mining import (
    EdgeMiner,
    BarEdgeData,
    EdgeMiningResult,
)

__all__ = [
    # Matrix serialization
    "snapshot_to_row",
    "snapshots_to_matrix",
    "compute_session_start_atr",
    "get_session_bar_count",
    # Future truth
    "BarMetadata",
    "SessionDataset",
    "generate_session_id",
    "compute_bar_metadata",
    "compute_session_dataset",
    # Storage
    "get_edge_finder_path",
    "get_sessions_path",
    "save_session_dataset",
    "load_session_dataset",
    "list_session_files",
    "get_session_stats",
    # Generator
    "generate_session_dataset",
    "generate_all_sessions",
    "generate_test_dataset",
    "iter_sessions",
    "get_available_pairs",
    # Vector index
    "LatentVectorIndex",
    "IndexedVector",
    "SearchResult",
    # Inference
    "EdgeInferenceEngine",
    "EdgeProbabilities",
    # Mining
    "EdgeMiner",
    "BarEdgeData",
    "EdgeMiningResult",
]
