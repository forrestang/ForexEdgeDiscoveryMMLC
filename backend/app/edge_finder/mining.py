"""
Edge Miner for batch pattern scanning with dual-horizon metrics.

Performs FAISS-accelerated batch KNN search across all bars in a session
and computes edge metrics for both immediate (next bar) and structural
(session end) time horizons using vectorized numpy operations.
"""

from dataclasses import dataclass
from typing import Any, Literal, Optional
import numpy as np
from scipy.spatial.distance import cdist

from app.edge_finder.vector_index import LatentVectorIndex


@dataclass
class BarEdgeData:
    """
    Complete edge metrics for a single bar position.

    Contains both Horizon A (next bar) and Horizon B (session end) metrics.
    """

    bar_index: int

    # Horizon A: Next Bar (Immediate Edge)
    next_bar_win_rate: float  # % neighbors where next_bar_move > 0
    next_bar_avg_move: float  # Average next bar move in ATR
    next_bar_edge_score: float  # win_rate * avg_move

    # Horizon B: Session End (Structural Edge)
    session_bias: Literal["long", "short"]  # Majority direction
    session_win_rate: float  # % neighbors ending in bias direction
    session_avg_mfe: float  # Average MFE (absolute)
    session_avg_mae: float  # Average MAE (absolute)
    session_risk_reward: float  # avg_mfe / |avg_mae|
    session_edge_score: float  # direction × (probability × avg_MFE_of_dominant_side)

    # Metadata
    num_matches: int  # Number of unique session matches
    avg_distance: float  # Average distance to neighbors

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "bar_index": self.bar_index,
            "next_bar_win_rate": round(self.next_bar_win_rate, 4),
            "next_bar_avg_move": round(self.next_bar_avg_move, 4),
            "next_bar_edge_score": round(self.next_bar_edge_score, 4),
            "session_bias": self.session_bias,
            "session_win_rate": round(self.session_win_rate, 4),
            "session_avg_mfe": round(self.session_avg_mfe, 4),
            "session_avg_mae": round(self.session_avg_mae, 4),
            "session_risk_reward": round(self.session_risk_reward, 4),
            "session_edge_score": round(self.session_edge_score, 4),
            "num_matches": self.num_matches,
            "avg_distance": round(self.avg_distance, 4),
        }


@dataclass
class EdgeMiningResult:
    """
    Complete mining result for all bars in a session.

    Provides both graph-ready data (simplified) and full edge table.
    """

    graph_data: list[dict]  # [{bar_index, session_score, next_bar_score}, ...]
    edge_table: list[BarEdgeData]  # Full metrics for all bars

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "graph_data": self.graph_data,
            "edge_table": [e.to_dict() for e in self.edge_table],
        }


class EdgeMiner:
    """
    Batch pattern scanner with FAISS-accelerated KNN and dual-horizon metrics.

    Processes all bars in a session simultaneously using vectorized operations.
    Primary backend is FAISS; falls back to scipy if FAISS unavailable.
    """

    def __init__(
        self,
        index: LatentVectorIndex,
        k_neighbors: int = 50,
        use_faiss: bool = True,
    ):
        """
        Initialize the EdgeMiner.

        Args:
            index: Pre-loaded LatentVectorIndex with vectors and metadata
            k_neighbors: Number of neighbors per query bar
            use_faiss: Attempt to use FAISS (fallback to scipy if unavailable)
        """
        self.index = index
        self.k_neighbors = k_neighbors
        self.use_faiss = use_faiss

        # FAISS index (lazy initialized)
        self._faiss_index: Optional[Any] = None
        self._faiss_available: bool = False

        # Check FAISS availability and build index
        if use_faiss:
            self._init_faiss()

    def _init_faiss(self) -> None:
        """Initialize FAISS index if available."""
        try:
            import faiss

            # Build L2 (Euclidean) index from existing vectors
            vectors = self.index.vectors.astype(np.float32)
            if len(vectors) == 0:
                self._faiss_available = False
                return

            latent_dim = vectors.shape[1]

            # Ensure vectors are contiguous
            vectors = np.ascontiguousarray(vectors)

            # Use IndexFlatL2 for exact search
            self._faiss_index = faiss.IndexFlatL2(latent_dim)
            self._faiss_index.add(vectors)

            self._faiss_available = True

        except ImportError:
            self._faiss_available = False

    def mine_from_full_matrix(
        self,
        full_matrix: np.ndarray,
    ) -> EdgeMiningResult:
        """
        Mine edge probabilities from a full session matrix.

        This is the main entry point for mining a complete session.

        Args:
            full_matrix: [num_bars, 20] complete session waveform matrix

        Returns:
            EdgeMiningResult with graph_data and edge_table
        """
        # 1. Encode all prefix matrices to latent vectors
        query_vectors = self._encode_all_prefixes(full_matrix)

        # 2. Batch KNN search
        if self._faiss_available and self.use_faiss:
            distances, indices = self._batch_search_faiss(query_vectors)
        else:
            distances, indices = self._batch_search_scipy(query_vectors)

        # 3. Filter to unique sessions per query bar
        filtered_distances, filtered_indices, valid_counts = (
            self._filter_unique_sessions(distances, indices)
        )

        # 4. Compute all metrics using vectorized operations
        edge_table = self._compute_metrics_vectorized(
            filtered_indices, filtered_distances, valid_counts
        )

        # 5. Build graph data (simplified for charting)
        graph_data = [
            {
                "bar_index": edge.bar_index,
                "session_score": edge.session_edge_score,
                "next_bar_score": edge.next_bar_edge_score,
            }
            for edge in edge_table
        ]

        return EdgeMiningResult(
            graph_data=graph_data,
            edge_table=edge_table,
        )

    def _encode_all_prefixes(
        self,
        full_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Encode all prefix matrices to latent vectors.

        Args:
            full_matrix: [num_bars, 20] session matrix

        Returns:
            query_vectors: [num_bars, latent_dim] encoded vectors
        """
        num_bars = full_matrix.shape[0]
        latent_dim = self.index.latent_dim

        query_vectors = np.zeros((num_bars, latent_dim), dtype=np.float32)

        # Encode each prefix
        for i in range(num_bars):
            prefix = full_matrix[: i + 1]  # [1:i+1, 20]
            query_vectors[i] = self.index.encode_matrix(prefix)

        return query_vectors

    def _batch_search_faiss(
        self,
        query_vectors: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Batch KNN search using FAISS.

        Args:
            query_vectors: [num_queries, latent_dim] float32

        Returns:
            distances: [num_queries, k] L2 distances
            indices: [num_queries, k] neighbor indices
        """
        # Ensure float32 and contiguous
        queries = np.ascontiguousarray(query_vectors, dtype=np.float32)

        # FAISS batch search - returns squared L2 distances
        squared_distances, indices = self._faiss_index.search(
            queries, self.k_neighbors
        )

        # Convert squared distances to regular distances
        distances = np.sqrt(np.maximum(squared_distances, 0))

        return distances, indices

    def _batch_search_scipy(
        self,
        query_vectors: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Batch KNN search using scipy cdist (fallback).

        Args:
            query_vectors: [num_queries, latent_dim]

        Returns:
            distances: [num_queries, k]
            indices: [num_queries, k]
        """
        # Compute all pairwise distances
        # Result: [num_queries, N]
        all_distances = cdist(query_vectors, self.index.vectors, metric="euclidean")

        # Get top-k indices for each query
        k = min(self.k_neighbors, all_distances.shape[1])

        # np.argpartition is O(N) vs O(N log N) for full sort
        indices = np.argpartition(all_distances, k, axis=1)[:, :k]

        # Gather corresponding distances
        row_indices = np.arange(all_distances.shape[0])[:, None]
        distances = all_distances[row_indices, indices]

        # Sort within each row's k elements
        sorted_order = np.argsort(distances, axis=1)
        indices = np.take_along_axis(indices, sorted_order, axis=1)
        distances = np.take_along_axis(distances, sorted_order, axis=1)

        return distances.astype(np.float32), indices.astype(np.int64)

    def _filter_unique_sessions(
        self,
        distances: np.ndarray,
        indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter to unique sessions per query bar.

        For each query bar, keep only the closest match per unique session_id.
        This prevents a single volatile session from dominating results.

        Args:
            distances: [num_queries, k] raw distances
            indices: [num_queries, k] raw indices

        Returns:
            filtered_distances: [num_queries, max_unique] padded distances (-1 for invalid)
            filtered_indices: [num_queries, max_unique] padded indices (-1 for invalid)
            valid_counts: [num_queries] number of valid matches per query
        """
        num_queries, k = indices.shape

        # Convert session_ids list to numpy array for vectorized lookup
        session_ids_array = np.array(self.index.session_ids, dtype=object)

        # Get session_ids for all neighbors: [num_queries, k]
        # Handle potential -1 indices from FAISS (when fewer than k vectors exist)
        safe_indices_for_lookup = np.maximum(indices, 0)
        neighbor_session_ids = session_ids_array[safe_indices_for_lookup]

        # Prepare output arrays
        max_unique = k  # Upper bound
        filtered_distances = np.full(
            (num_queries, max_unique), -1.0, dtype=np.float32
        )
        filtered_indices = np.full((num_queries, max_unique), -1, dtype=np.int64)
        valid_counts = np.zeros(num_queries, dtype=np.int32)

        for q in range(num_queries):
            seen_sessions: set = set()
            write_idx = 0

            for n in range(k):
                # Skip invalid indices
                if indices[q, n] < 0:
                    continue

                session_id = neighbor_session_ids[q, n]
                if session_id not in seen_sessions:
                    seen_sessions.add(session_id)
                    filtered_distances[q, write_idx] = distances[q, n]
                    filtered_indices[q, write_idx] = indices[q, n]
                    write_idx += 1

            valid_counts[q] = write_idx

        return filtered_distances, filtered_indices, valid_counts

    def _compute_metrics_vectorized(
        self,
        indices: np.ndarray,
        distances: np.ndarray,
        valid_counts: np.ndarray,
    ) -> list[BarEdgeData]:
        """
        Compute all edge metrics using vectorized numpy operations.

        Args:
            indices: [num_queries, max_matches] neighbor indices, -1 for invalid
            distances: [num_queries, max_matches] neighbor distances, -1 for invalid
            valid_counts: [num_queries] number of valid matches per query

        Returns:
            List of BarEdgeData, one per query bar
        """
        num_queries, max_matches = indices.shape

        # Create validity mask: [num_queries, max_matches]
        valid_mask = indices >= 0

        # Use np.maximum to avoid negative indexing for invalid entries
        safe_indices = np.maximum(indices, 0)

        # Gather ground truth data for all neighbors
        # Shape: [num_queries, max_matches]
        next_bar_moves = self.index.next_bar_moves[safe_indices]
        session_drifts = self.index.session_drifts[safe_indices]
        maes = self.index.maes[safe_indices]
        mfes = self.index.mfes[safe_indices]

        # Zero out invalid entries
        next_bar_moves = np.where(valid_mask, next_bar_moves, 0.0)
        session_drifts = np.where(valid_mask, session_drifts, 0.0)
        maes = np.where(valid_mask, maes, 0.0)
        mfes = np.where(valid_mask, mfes, 0.0)
        distances_masked = np.where(valid_mask, distances, 0.0)

        # Count valid matches per query
        valid_float = valid_mask.astype(np.float32)
        counts = valid_mask.sum(axis=1).astype(np.float32)
        counts_safe = np.maximum(counts, 1.0)  # Avoid division by zero

        # ========================================
        # HORIZON A: Next Bar Metrics (Conditional Expectancy)
        # ========================================
        # Same formula as Session: direction × (probability × avg_MFE_of_dominant_side)

        # 1. Determine Bias: next_bar_move > 0 = bull, < 0 = bear
        nb_bull_mask = (next_bar_moves > 0).astype(np.float32) * valid_float
        nb_bear_mask = (next_bar_moves < 0).astype(np.float32) * valid_float

        nb_bull_count = nb_bull_mask.sum(axis=1)
        nb_bear_count = nb_bear_mask.sum(axis=1)

        # 2. Select Dominant Side (ties = no edge, score will be 0)
        nb_bias_is_long = nb_bull_count > nb_bear_count
        nb_bias_is_short = nb_bear_count > nb_bull_count

        # Direction: +1 for long, -1 for short, 0 for tie
        nb_direction_sign = np.where(
            nb_bias_is_long, 1.0, np.where(nb_bias_is_short, -1.0, 0.0)
        )

        # Probability = dominant_count / total
        nb_dominant_count = np.where(nb_bias_is_long, nb_bull_count, nb_bear_count)
        nb_probability = nb_dominant_count / counts_safe

        # 3. MFE = Average |next_bar_move| of ONLY dominant-side neighbors
        abs_next_bar_moves = np.abs(next_bar_moves)

        nb_bull_mfe_sum = (abs_next_bar_moves * nb_bull_mask).sum(axis=1)
        nb_bear_mfe_sum = (abs_next_bar_moves * nb_bear_mask).sum(axis=1)

        nb_dominant_count_safe = np.maximum(nb_dominant_count, 1.0)

        nb_raw_ev = np.where(
            nb_bias_is_long,
            nb_bull_mfe_sum / nb_dominant_count_safe,
            nb_bear_mfe_sum / nb_dominant_count_safe,
        )

        # 4. Final Score: direction × (probability × raw_EV)
        next_bar_edge_score = nb_direction_sign * (nb_probability * nb_raw_ev)

        # Win rate = probability for display consistency
        next_bar_win_rate = nb_probability

        # Keep avg_move for display (uses all neighbors)
        next_bar_avg_move = next_bar_moves.sum(axis=1) / counts_safe

        # ========================================
        # HORIZON B: Session End Metrics (Conditional Expectancy)
        # ========================================
        # Instead of blending wins/losses, we focus on the dominant direction:
        # Edge Score = direction × (probability × avg_MFE_of_dominant_side)

        # 1. Determine Bias (The Vote)
        # Bull = session_drift > 0, Bear = session_drift < 0
        bull_mask = (session_drifts > 0).astype(np.float32) * valid_float
        bear_mask = (session_drifts < 0).astype(np.float32) * valid_float

        bull_count = bull_mask.sum(axis=1)
        bear_count = bear_mask.sum(axis=1)

        # 2. Select Dominant Side (ties = no edge, score will be 0)
        session_bias_is_long = bull_count > bear_count
        session_bias_is_short = bear_count > bull_count

        # Direction: +1 for long, -1 for short, 0 for tie
        direction_sign = np.where(
            session_bias_is_long, 1.0, np.where(session_bias_is_short, -1.0, 0.0)
        )

        # Probability = dominant_count / total
        dominant_count = np.where(session_bias_is_long, bull_count, bear_count)
        probability = dominant_count / counts_safe

        # 3. Raw_EV = Average MFE of ONLY dominant-side neighbors
        abs_mfes = np.abs(mfes)
        abs_maes = np.abs(maes)

        bull_mfe_sum = (abs_mfes * bull_mask).sum(axis=1)
        bear_mfe_sum = (abs_mfes * bear_mask).sum(axis=1)

        # Avoid division by zero for dominant count
        dominant_count_safe = np.maximum(dominant_count, 1.0)

        raw_ev = np.where(
            session_bias_is_long,
            bull_mfe_sum / dominant_count_safe,
            bear_mfe_sum / dominant_count_safe,
        )

        # 4. Final Score: direction × (probability × raw_EV)
        session_edge_score = direction_sign * (probability * raw_ev)

        # Keep win_rate = probability for display
        session_win_rate = probability

        # Keep avg_mfe/avg_mae for R:R display (uses all neighbors)
        avg_mfe = (abs_mfes * valid_float).sum(axis=1) / counts_safe
        avg_mae = (abs_maes * valid_float).sum(axis=1) / counts_safe

        # Risk/Reward: MFE / MAE
        risk_reward = np.where(
            avg_mae > 1e-6,
            avg_mfe / avg_mae,
            0.0,
        )

        # ========================================
        # Metadata
        # ========================================

        avg_distance = (distances_masked * valid_float).sum(axis=1) / counts_safe

        # ========================================
        # Build BarEdgeData list
        # ========================================

        results = []
        for i in range(num_queries):
            results.append(
                BarEdgeData(
                    bar_index=i,
                    next_bar_win_rate=float(next_bar_win_rate[i]),
                    next_bar_avg_move=float(next_bar_avg_move[i]),
                    next_bar_edge_score=float(next_bar_edge_score[i]),
                    # Use >= so ties default to "long" label (score will be 0 anyway)
                    session_bias="long" if bull_count[i] >= bear_count[i] else "short",
                    session_win_rate=float(session_win_rate[i]),
                    session_avg_mfe=float(avg_mfe[i]),
                    session_avg_mae=float(avg_mae[i]),
                    session_risk_reward=float(risk_reward[i]),
                    session_edge_score=float(session_edge_score[i]),
                    num_matches=int(valid_counts[i]),
                    avg_distance=float(avg_distance[i]),
                )
            )

        return results
