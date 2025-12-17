"""
Latent Vector Index for pattern similarity search.

Loads session matrices, encodes them to latent vectors using a trained VAE,
and provides KNN search capabilities for finding similar historical patterns.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from scipy.spatial.distance import cdist

from app.edge_finder.storage import (
    get_sessions_path,
    get_models_path,
    get_vectors_path,
    list_session_files,
)
from app.edge_finder.model.vae import WaveformVAE
from app.edge_finder.model.config import TrainingConfig


@dataclass
class IndexedVector:
    """
    A single indexed latent vector with its metadata.
    """
    session_id: str
    bar_index: int
    latent_vector: np.ndarray  # [latent_dim]
    # Ground truth outcomes (for edge probability computation)
    next_bar_move: float
    session_drift: float
    mae: float
    mfe: float
    session_progress: float


@dataclass
class SearchResult:
    """
    A single search result from KNN query.
    """
    session_id: str
    bar_index: int
    distance: float
    next_bar_move: float
    session_drift: float
    mae: float
    mfe: float
    session_progress: float


class LatentVectorIndex:
    """
    In-memory index of latent vectors for KNN similarity search.

    Loads all session datasets, encodes their matrices using a trained VAE,
    and provides efficient search via scipy cdist.
    """

    def __init__(
        self,
        model_name: str = "vae_test",
        working_directory: Optional[Path] = None,
        device: str = "auto",
    ):
        """
        Initialize the vector index.

        Args:
            model_name: Name of the trained VAE model (without extension)
            working_directory: Optional override for data directory
            device: Device for model inference ("auto", "cuda", "cpu")
        """
        self.model_name = model_name
        self.working_directory = working_directory
        self.device = self._get_device(device)

        # Index storage
        self.vectors: np.ndarray = np.array([])  # [num_vectors, latent_dim]
        self.session_ids: list[str] = []
        self.bar_indices: list[int] = []
        self.next_bar_moves: np.ndarray = np.array([])
        self.session_drifts: np.ndarray = np.array([])
        self.maes: np.ndarray = np.array([])
        self.mfes: np.ndarray = np.array([])
        self.session_progress: np.ndarray = np.array([])

        # Model
        self.model: Optional[WaveformVAE] = None
        self.config: Optional[TrainingConfig] = None

        # Track loaded state
        self._is_loaded = False

    def _get_device(self, device: str) -> str:
        """Resolve 'auto' device to actual device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def load_model(self) -> None:
        """Load the trained VAE model."""
        models_path = get_models_path(self.working_directory)

        # Load config
        config_path = models_path / f"{self.model_name}_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Model config not found: {config_path}")
        self.config = TrainingConfig.load(config_path)

        # Create model
        self.model = WaveformVAE(
            input_dim=self.config.input_channels,
            hidden_dim=self.config.hidden_dim,
            latent_dim=self.config.latent_dim,
            num_layers=self.config.num_layers,
            bidirectional=self.config.bidirectional,
            dropout=self.config.dropout,
        )

        # Load weights (prefer best model if available)
        best_path = models_path / f"{self.model_name}_best.pt"
        model_path = models_path / f"{self.model_name}.pt"

        weights_path = best_path if best_path.exists() else model_path
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")

        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)

        # Handle checkpoint format (contains model_state_dict) vs raw state_dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def encode_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """
        Encode a waveform matrix to latent vector.

        Args:
            matrix: [seq_len, 20] waveform matrix

        Returns:
            latent: [latent_dim] latent vector
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        with torch.no_grad():
            # Add batch dimension: [1, seq_len, 20]
            x = torch.tensor(matrix, dtype=torch.float32, device=self.device).unsqueeze(0)
            lengths = torch.tensor([matrix.shape[0]], device=self.device)

            # Encode to latent (returns mean, no sampling)
            z = self.model.encode(x, lengths)

            return z.cpu().numpy().squeeze(0)

    def build_index(
        self,
        pair: Optional[str] = None,
        session_type: Optional[str] = None,
        timeframe: Optional[str] = None,
        progress_callback: Optional[callable] = None,
    ) -> int:
        """
        Build the vector index from all matching session datasets.

        Args:
            pair: Filter by currency pair (e.g., "EURUSD")
            session_type: Filter by session type (e.g., "ny")
            timeframe: Filter by timeframe (e.g., "M10")
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            Number of vectors indexed
        """
        if self.model is None:
            self.load_model()

        # Get list of sessions to index
        session_ids = list_session_files(
            working_directory=self.working_directory,
            pair=pair,
            session_type=session_type,
            timeframe=timeframe,
        )

        if not session_ids:
            return 0

        sessions_path = get_sessions_path(self.working_directory)

        # Collect all data
        all_vectors = []
        all_session_ids = []
        all_bar_indices = []
        all_next_bar_moves = []
        all_session_drifts = []
        all_maes = []
        all_mfes = []
        all_session_progress = []

        for idx, session_id in enumerate(session_ids):
            if progress_callback:
                progress_callback(idx + 1, len(session_ids))

            # Load session data
            file_path = sessions_path / f"{session_id}.npz"
            if not file_path.exists():
                continue

            data = np.load(file_path, allow_pickle=True)
            matrix = data["matrix"]
            n_bars = int(data["total_bars"])

            # Encode full matrix
            latent = self.encode_matrix(matrix)

            # Index each bar position by encoding the prefix up to that bar
            for bar_idx in range(n_bars):
                # Encode prefix matrix [0:bar_idx+1]
                prefix_matrix = matrix[:bar_idx + 1]
                bar_latent = self.encode_matrix(prefix_matrix)

                all_vectors.append(bar_latent)
                all_session_ids.append(session_id)
                all_bar_indices.append(bar_idx)
                all_next_bar_moves.append(float(data["next_bar_moves"][bar_idx]))
                all_session_drifts.append(float(data["session_drifts"][bar_idx]))
                all_maes.append(float(data["maes"][bar_idx]))
                all_mfes.append(float(data["mfes"][bar_idx]))
                all_session_progress.append(float(data["session_progress"][bar_idx]))

        # Convert to arrays
        self.vectors = np.array(all_vectors, dtype=np.float32)
        self.session_ids = all_session_ids
        self.bar_indices = all_bar_indices
        self.next_bar_moves = np.array(all_next_bar_moves, dtype=np.float32)
        self.session_drifts = np.array(all_session_drifts, dtype=np.float32)
        self.maes = np.array(all_maes, dtype=np.float32)
        self.mfes = np.array(all_mfes, dtype=np.float32)
        self.session_progress = np.array(all_session_progress, dtype=np.float32)

        self._is_loaded = True
        return len(self.vectors)

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 500,
        metric: str = "euclidean",
    ) -> list[SearchResult]:
        """
        Find K nearest neighbors to query vector.

        Args:
            query_vector: [latent_dim] query latent vector
            k: Number of neighbors to return
            metric: Distance metric ("euclidean", "cosine", etc.)

        Returns:
            List of SearchResult sorted by distance (ascending)
        """
        if not self._is_loaded or len(self.vectors) == 0:
            raise RuntimeError("Index not built. Call build_index() first.")

        # Compute distances
        query = query_vector.reshape(1, -1)
        distances = cdist(query, self.vectors, metric=metric).squeeze(0)

        # Get top-k indices
        k = min(k, len(distances))
        top_k_indices = np.argpartition(distances, k)[:k]
        top_k_indices = top_k_indices[np.argsort(distances[top_k_indices])]

        # Build results
        results = []
        for idx in top_k_indices:
            results.append(SearchResult(
                session_id=self.session_ids[idx],
                bar_index=self.bar_indices[idx],
                distance=float(distances[idx]),
                next_bar_move=float(self.next_bar_moves[idx]),
                session_drift=float(self.session_drifts[idx]),
                mae=float(self.maes[idx]),
                mfe=float(self.mfes[idx]),
                session_progress=float(self.session_progress[idx]),
            ))

        return results

    def search_unique_sessions(
        self,
        query_vector: np.ndarray,
        k: int = 500,
        metric: str = "euclidean",
    ) -> list[SearchResult]:
        """
        Find K nearest neighbors, filtered to best match per session.

        This prevents bias from a single volatile session dominating results.
        For each unique session, only the closest matching bar is returned.

        Args:
            query_vector: [latent_dim] query latent vector
            k: Maximum number of unique sessions to return
            metric: Distance metric

        Returns:
            List of SearchResult (one per unique session), sorted by distance
        """
        if not self._is_loaded or len(self.vectors) == 0:
            raise RuntimeError("Index not built. Call build_index() first.")

        # Compute all distances
        query = query_vector.reshape(1, -1)
        distances = cdist(query, self.vectors, metric=metric).squeeze(0)

        # Sort by distance
        sorted_indices = np.argsort(distances)

        # Filter to best match per session
        seen_sessions: set[str] = set()
        results = []

        for idx in sorted_indices:
            session_id = self.session_ids[idx]

            if session_id in seen_sessions:
                continue

            seen_sessions.add(session_id)
            results.append(SearchResult(
                session_id=session_id,
                bar_index=self.bar_indices[idx],
                distance=float(distances[idx]),
                next_bar_move=float(self.next_bar_moves[idx]),
                session_drift=float(self.session_drifts[idx]),
                mae=float(self.maes[idx]),
                mfe=float(self.mfes[idx]),
                session_progress=float(self.session_progress[idx]),
            ))

            if len(results) >= k:
                break

        return results

    def save_index(self, name: str = "latent_index") -> Path:
        """
        Save the built index to disk for faster loading.

        Args:
            name: Name for the saved index files

        Returns:
            Path to saved index
        """
        if not self._is_loaded:
            raise RuntimeError("Index not built. Call build_index() first.")

        vectors_path = get_vectors_path(self.working_directory)
        index_path = vectors_path / f"{name}.npz"

        np.savez_compressed(
            index_path,
            vectors=self.vectors,
            session_ids=np.array(self.session_ids, dtype=object),
            bar_indices=np.array(self.bar_indices, dtype=np.int32),
            next_bar_moves=self.next_bar_moves,
            session_drifts=self.session_drifts,
            maes=self.maes,
            mfes=self.mfes,
            session_progress=self.session_progress,
            model_name=self.model_name,
        )

        return index_path

    def load_index(self, name: str = "latent_index") -> bool:
        """
        Load a previously saved index from disk.

        Args:
            name: Name of the saved index

        Returns:
            True if loaded successfully, False if not found
        """
        vectors_path = get_vectors_path(self.working_directory)
        index_path = vectors_path / f"{name}.npz"

        if not index_path.exists():
            return False

        data = np.load(index_path, allow_pickle=True)

        self.vectors = data["vectors"]
        self.session_ids = list(data["session_ids"])
        self.bar_indices = list(data["bar_indices"])
        self.next_bar_moves = data["next_bar_moves"]
        self.session_drifts = data["session_drifts"]
        self.maes = data["maes"]
        self.mfes = data["mfes"]
        self.session_progress = data["session_progress"]

        # Load model if needed
        saved_model_name = str(data["model_name"])
        if self.model_name != saved_model_name:
            self.model_name = saved_model_name

        if self.model is None:
            self.load_model()

        self._is_loaded = True
        return True

    @property
    def num_vectors(self) -> int:
        """Number of indexed vectors."""
        return len(self.vectors) if self._is_loaded else 0

    @property
    def num_sessions(self) -> int:
        """Number of unique sessions in index."""
        return len(set(self.session_ids)) if self._is_loaded else 0

    @property
    def latent_dim(self) -> int:
        """Latent dimension of indexed vectors."""
        if self._is_loaded and len(self.vectors) > 0:
            return self.vectors.shape[1]
        if self.config:
            return self.config.latent_dim
        return 32  # Default
