"""
Edge Inference Engine for computing edge probabilities.

Encodes current waveform patterns, finds similar historical patterns,
and aggregates outcomes into edge probability statistics.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np

from app.edge_finder.vector_index import LatentVectorIndex, SearchResult


@dataclass
class MatchDetail:
    """
    Details of a single matched historical pattern.

    Used to display matched sessions in the UI for sanity checking.
    """
    session_id: str           # e.g., "EURUSD_2024-01-15_london_M5"
    bar_index: int            # Bar position in the matched session
    distance: float           # Similarity distance (lower = better)
    next_bar_move: float      # What happened next bar (ATR normalized)
    session_drift: float      # Drift to session end (ATR normalized)
    mae: float                # Maximum adverse excursion
    mfe: float                # Maximum favorable excursion
    session_progress: float   # Progress through session (0-1)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "bar_index": self.bar_index,
            "distance": round(self.distance, 4),
            "next_bar_move": round(self.next_bar_move, 4),
            "session_drift": round(self.session_drift, 4),
            "mae": round(self.mae, 4),
            "mfe": round(self.mfe, 4),
            "session_progress": round(self.session_progress, 4),
        }


@dataclass
class EdgeProbabilities:
    """
    Aggregated edge probability statistics from similar historical patterns.

    All metrics are computed from the K nearest historical patterns,
    filtered to unique sessions to prevent single-session bias.
    """
    # Sample info
    num_matches: int              # Number of unique sessions matched
    avg_distance: float           # Average distance to matches

    # Next bar edge (immediate)
    next_bar_up_pct: float        # Percentage of matches where next bar was up
    next_bar_avg_move: float      # Average next bar move (ATR normalized)
    next_bar_std_move: float      # Standard deviation of next bar moves

    # Session-level edge
    session_up_pct: float         # Percentage of sessions that ended higher
    session_avg_drift: float      # Average session drift (ATR normalized)
    session_std_drift: float      # Standard deviation of session drift

    # Risk metrics (MAE - Maximum Adverse Excursion)
    avg_mae: float                # Average MAE (worst drawdown)
    mae_p25: float                # 25th percentile MAE
    mae_p50: float                # Median MAE
    mae_p75: float                # 75th percentile MAE
    mae_p95: float                # 95th percentile MAE (worst case)

    # Reward metrics (MFE - Maximum Favorable Excursion)
    avg_mfe: float                # Average MFE (best runup)
    mfe_p25: float                # 25th percentile MFE
    mfe_p50: float                # Median MFE
    mfe_p75: float                # 75th percentile MFE
    mfe_p95: float                # 95th percentile MFE

    # Risk/reward ratio
    risk_reward_ratio: float      # avg_mfe / abs(avg_mae) if mae != 0

    # Session progress context
    avg_session_progress: float   # Average progress through session

    # Match quality
    top_10_avg_distance: float    # Average distance of top 10 matches
    match_session_ids: list[str] = field(default_factory=list)  # Top match session IDs

    # Detailed match info for UI display
    top_matches: list[MatchDetail] = field(default_factory=list)  # Top K matches with full details

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "num_matches": self.num_matches,
            "avg_distance": round(self.avg_distance, 4),
            "next_bar_up_pct": round(self.next_bar_up_pct, 4),
            "next_bar_avg_move": round(self.next_bar_avg_move, 4),
            "next_bar_std_move": round(self.next_bar_std_move, 4),
            "session_up_pct": round(self.session_up_pct, 4),
            "session_avg_drift": round(self.session_avg_drift, 4),
            "session_std_drift": round(self.session_std_drift, 4),
            "avg_mae": round(self.avg_mae, 4),
            "mae_p25": round(self.mae_p25, 4),
            "mae_p50": round(self.mae_p50, 4),
            "mae_p75": round(self.mae_p75, 4),
            "mae_p95": round(self.mae_p95, 4),
            "avg_mfe": round(self.avg_mfe, 4),
            "mfe_p25": round(self.mfe_p25, 4),
            "mfe_p50": round(self.mfe_p50, 4),
            "mfe_p75": round(self.mfe_p75, 4),
            "mfe_p95": round(self.mfe_p95, 4),
            "risk_reward_ratio": round(self.risk_reward_ratio, 4),
            "avg_session_progress": round(self.avg_session_progress, 4),
            "top_10_avg_distance": round(self.top_10_avg_distance, 4),
            "match_session_ids": self.match_session_ids[:10],  # Top 10 only
            "top_matches": [m.to_dict() for m in self.top_matches[:50]],  # Top 50 matches
        }


class EdgeInferenceEngine:
    """
    Engine for computing edge probabilities from waveform patterns.

    Encodes current waveform state, finds similar historical patterns,
    and aggregates their outcomes into actionable edge statistics.
    """

    def __init__(
        self,
        model_name: str = "vae_test",
        working_directory: Optional[Path] = None,
        k_neighbors: int = 500,
        device: str = "auto",
    ):
        """
        Initialize the inference engine.

        Args:
            model_name: Name of the trained VAE model
            working_directory: Optional override for data directory
            k_neighbors: Number of neighbors for KNN search
            device: Device for model inference
        """
        self.model_name = model_name
        self.working_directory = working_directory
        self.k_neighbors = k_neighbors
        self.device = device

        # Vector index
        self.index = LatentVectorIndex(
            model_name=model_name,
            working_directory=working_directory,
            device=device,
        )

        self._is_ready = False

    def initialize(
        self,
        load_saved_index: bool = True,
        index_name: str = "latent_index",
        pair: Optional[str] = None,
        session_type: Optional[str] = None,
        timeframe: Optional[str] = None,
        progress_callback: Optional[callable] = None,
    ) -> int:
        """
        Initialize the engine by loading or building the vector index.

        Args:
            load_saved_index: Try to load saved index first
            index_name: Name of saved index to load
            pair: Filter for building index
            session_type: Filter for building index
            timeframe: Filter for building index
            progress_callback: Callback for build progress

        Returns:
            Number of vectors in index
        """
        # Try to load saved index
        if load_saved_index and self.index.load_index(index_name):
            self._is_ready = True
            return self.index.num_vectors

        # Build index from scratch
        self.index.load_model()
        num_vectors = self.index.build_index(
            pair=pair,
            session_type=session_type,
            timeframe=timeframe,
            progress_callback=progress_callback,
        )

        self._is_ready = num_vectors > 0
        return num_vectors

    def compute_edge(
        self,
        matrix: np.ndarray,
        k: Optional[int] = None,
        unique_sessions: bool = True,
    ) -> EdgeProbabilities:
        """
        Compute edge probabilities for a waveform matrix.

        Args:
            matrix: [seq_len, 20] current waveform matrix
            k: Number of neighbors (defaults to self.k_neighbors)
            unique_sessions: Filter to one match per session (anti-bias)

        Returns:
            EdgeProbabilities with aggregated statistics
        """
        if not self._is_ready:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        k = k or self.k_neighbors

        # Encode current pattern
        query_vector = self.index.encode_matrix(matrix)

        # Find similar patterns
        if unique_sessions:
            results = self.index.search_unique_sessions(query_vector, k=k)
        else:
            results = self.index.search(query_vector, k=k)

        # Aggregate outcomes
        return self._aggregate_results(results)

    def compute_edge_from_latent(
        self,
        latent_vector: np.ndarray,
        k: Optional[int] = None,
        unique_sessions: bool = True,
    ) -> EdgeProbabilities:
        """
        Compute edge probabilities from a pre-encoded latent vector.

        Args:
            latent_vector: [latent_dim] latent vector
            k: Number of neighbors
            unique_sessions: Filter to one match per session

        Returns:
            EdgeProbabilities with aggregated statistics
        """
        if not self._is_ready:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        k = k or self.k_neighbors

        if unique_sessions:
            results = self.index.search_unique_sessions(latent_vector, k=k)
        else:
            results = self.index.search(latent_vector, k=k)

        return self._aggregate_results(results)

    def _aggregate_results(self, results: list[SearchResult]) -> EdgeProbabilities:
        """
        Aggregate search results into edge probability statistics.

        Args:
            results: List of SearchResult from KNN search

        Returns:
            EdgeProbabilities with aggregated statistics
        """
        if not results:
            return self._empty_edge_probabilities()

        # Extract arrays
        distances = np.array([r.distance for r in results])
        next_bar_moves = np.array([r.next_bar_move for r in results])
        session_drifts = np.array([r.session_drift for r in results])
        maes = np.array([r.mae for r in results])
        mfes = np.array([r.mfe for r in results])
        session_progress = np.array([r.session_progress for r in results])
        session_ids = [r.session_id for r in results]

        # Next bar statistics
        next_bar_up_pct = float(np.mean(next_bar_moves > 0))
        next_bar_avg = float(np.mean(next_bar_moves))
        next_bar_std = float(np.std(next_bar_moves))

        # Session statistics
        session_up_pct = float(np.mean(session_drifts > 0))
        session_avg = float(np.mean(session_drifts))
        session_std = float(np.std(session_drifts))

        # MAE percentiles (these are typically negative for long positions)
        mae_percentiles = np.percentile(maes, [25, 50, 75, 95])

        # MFE percentiles
        mfe_percentiles = np.percentile(mfes, [25, 50, 75, 95])

        # Risk/reward ratio
        avg_mae = float(np.mean(maes))
        avg_mfe = float(np.mean(mfes))
        risk_reward = avg_mfe / abs(avg_mae) if abs(avg_mae) > 1e-6 else 0.0

        # Distance metrics
        avg_distance = float(np.mean(distances))
        top_10_distance = float(np.mean(distances[:min(10, len(distances))]))

        # Create detailed match info for top matches
        top_matches = [
            MatchDetail(
                session_id=r.session_id,
                bar_index=r.bar_index,
                distance=r.distance,
                next_bar_move=r.next_bar_move,
                session_drift=r.session_drift,
                mae=r.mae,
                mfe=r.mfe,
                session_progress=r.session_progress,
            )
            for r in results[:50]  # Top 50 matches
        ]

        return EdgeProbabilities(
            num_matches=len(results),
            avg_distance=avg_distance,
            next_bar_up_pct=next_bar_up_pct,
            next_bar_avg_move=next_bar_avg,
            next_bar_std_move=next_bar_std,
            session_up_pct=session_up_pct,
            session_avg_drift=session_avg,
            session_std_drift=session_std,
            avg_mae=avg_mae,
            mae_p25=float(mae_percentiles[0]),
            mae_p50=float(mae_percentiles[1]),
            mae_p75=float(mae_percentiles[2]),
            mae_p95=float(mae_percentiles[3]),
            avg_mfe=avg_mfe,
            mfe_p25=float(mfe_percentiles[0]),
            mfe_p50=float(mfe_percentiles[1]),
            mfe_p75=float(mfe_percentiles[2]),
            mfe_p95=float(mfe_percentiles[3]),
            risk_reward_ratio=risk_reward,
            avg_session_progress=float(np.mean(session_progress)),
            top_10_avg_distance=top_10_distance,
            match_session_ids=session_ids,
            top_matches=top_matches,
        )

    def _empty_edge_probabilities(self) -> EdgeProbabilities:
        """Return empty edge probabilities when no matches found."""
        return EdgeProbabilities(
            num_matches=0,
            avg_distance=0.0,
            next_bar_up_pct=0.5,
            next_bar_avg_move=0.0,
            next_bar_std_move=0.0,
            session_up_pct=0.5,
            session_avg_drift=0.0,
            session_std_drift=0.0,
            avg_mae=0.0,
            mae_p25=0.0,
            mae_p50=0.0,
            mae_p75=0.0,
            mae_p95=0.0,
            avg_mfe=0.0,
            mfe_p25=0.0,
            mfe_p50=0.0,
            mfe_p75=0.0,
            mfe_p95=0.0,
            risk_reward_ratio=0.0,
            avg_session_progress=0.0,
            top_10_avg_distance=0.0,
            match_session_ids=[],
            top_matches=[],
        )

    def save_index(self, name: str = "latent_index") -> Path:
        """Save the built index for faster future loading."""
        return self.index.save_index(name)

    @property
    def is_ready(self) -> bool:
        """Whether the engine is ready for inference."""
        return self._is_ready

    @property
    def num_indexed_vectors(self) -> int:
        """Number of vectors in the index."""
        return self.index.num_vectors

    @property
    def num_indexed_sessions(self) -> int:
        """Number of unique sessions in the index."""
        return self.index.num_sessions
