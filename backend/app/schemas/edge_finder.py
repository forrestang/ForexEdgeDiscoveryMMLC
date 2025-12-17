"""
Pydantic schemas for Edge Finder API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import date


# --- Session Generation ---

class GenerateSessionsRequest(BaseModel):
    """Request to generate session datasets from OHLC data."""
    working_directory: Optional[str] = None
    pair: Optional[str] = None  # e.g., "EURUSD", None = all pairs
    session_type: str = "ny"  # "asia", "london", "ny", "full_day"
    timeframe: str = "M10"  # "M5", "M10", "M15", "M30"
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    max_sessions: Optional[int] = None  # Limit for testing


class GenerateSessionsResponse(BaseModel):
    """Response from session generation."""
    status: str
    sessions_generated: int
    errors: list[str]
    message: str


class SessionStats(BaseModel):
    """Statistics about stored sessions."""
    total_sessions: int
    by_pair: dict[str, int]
    by_session_type: dict[str, int]
    by_timeframe: dict[str, int]


# --- Training ---

class TrainingRequest(BaseModel):
    """Request to start VAE training."""
    working_directory: Optional[str] = None
    model_name: str = "vae_default"
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    latent_dim: int = 32
    kl_weight: float = 0.1
    pair: Optional[str] = None  # Filter sessions
    session_type: Optional[str] = None
    timeframe: Optional[str] = None


class TrainingStatus(BaseModel):
    """Status of VAE training job."""
    is_training: bool
    epoch: int
    total_epochs: int
    train_loss: float
    val_loss: float
    best_loss: float
    progress: float  # 0.0 - 1.0
    message: str
    model_name: Optional[str] = None


class TrainingResponse(BaseModel):
    """Response from training start/stop."""
    status: str
    message: str


# --- Model Info ---

class ModelInfo(BaseModel):
    """Information about a trained model."""
    model_name: str
    latent_dim: int
    hidden_dim: int
    num_layers: int
    bidirectional: bool
    total_parameters: int
    trained_epochs: int
    best_loss: float
    created_at: Optional[str] = None


class ModelListResponse(BaseModel):
    """List of available models."""
    models: list[ModelInfo]


# --- Index ---

class BuildIndexRequest(BaseModel):
    """Request to build/rebuild the vector index."""
    working_directory: Optional[str] = None
    model_name: str = "vae_test"
    pair: Optional[str] = None
    session_type: Optional[str] = None
    timeframe: Optional[str] = None
    save_index: bool = True
    index_name: str = "latent_index"


class BuildIndexResponse(BaseModel):
    """Response from index building."""
    status: str
    num_vectors: int
    num_sessions: int
    message: str


class IndexStatus(BaseModel):
    """Status of the vector index."""
    is_loaded: bool
    num_vectors: int
    num_sessions: int
    latent_dim: int
    model_name: Optional[str] = None


# --- Inference ---

class MatchDetailResponse(BaseModel):
    """Details of a single matched historical pattern."""
    session_id: str
    bar_index: int
    distance: float
    next_bar_move: float
    session_drift: float
    mae: float
    mfe: float
    session_progress: float


class InferenceRequest(BaseModel):
    """Request for edge inference on current waveform state."""
    working_directory: Optional[str] = None
    matrix: list[list[float]]  # [seq_len, 20] waveform matrix
    k_neighbors: int = 500
    unique_sessions: bool = True


class EdgeProbabilitiesResponse(BaseModel):
    """Edge probability statistics from inference."""
    # Sample info
    num_matches: int
    avg_distance: float

    # Next bar edge
    next_bar_up_pct: float
    next_bar_avg_move: float
    next_bar_std_move: float

    # Session edge
    session_up_pct: float
    session_avg_drift: float
    session_std_drift: float

    # Risk metrics (MAE)
    avg_mae: float
    mae_p25: float
    mae_p50: float
    mae_p75: float
    mae_p95: float

    # Reward metrics (MFE)
    avg_mfe: float
    mfe_p25: float
    mfe_p50: float
    mfe_p75: float
    mfe_p95: float

    # Risk/reward
    risk_reward_ratio: float

    # Context
    avg_session_progress: float
    top_10_avg_distance: float

    # Detailed match info for UI display
    top_matches: list[MatchDetailResponse] = []


class InferenceResponse(BaseModel):
    """Response from edge inference."""
    status: str
    edge: Optional[EdgeProbabilitiesResponse] = None
    message: str


# --- Live Session Inference ---

class LiveInferenceRequest(BaseModel):
    """Request for inference on live/current session."""
    working_directory: Optional[str] = None
    pair: str
    session_type: str = "ny"
    timeframe: str = "M10"
    k_neighbors: int = 500
    unique_sessions: bool = True


class LiveInferenceResponse(BaseModel):
    """Response from live session inference."""
    status: str
    session_id: str
    current_bar: int
    total_bars: int
    edge: Optional[EdgeProbabilitiesResponse] = None
    message: str


# --- Chart Inference ---

class ChartInferenceRequest(BaseModel):
    """Request for inference on a specific chart (loads stored session)."""
    working_directory: Optional[str] = None
    pair: str
    date: str  # ISO format: "2022-01-03"
    session: str = "full_day"  # "asia", "london", "ny", "full_day"
    timeframe: str = "M10"
    bar_index: Optional[int] = None  # If None, use last bar
    k_neighbors: int = 500
    unique_sessions: bool = True


# --- Auto Setup ---

class AutoSetupRequest(BaseModel):
    """Request for automatic model/index setup."""
    working_directory: Optional[str] = None
    model_name: str = "vae_default"
    force_retrain: bool = False
    force_rebuild_index: bool = False
    # Training params (used if training needed)
    num_epochs: int = 100
    latent_dim: int = 32
    # Filter params
    pair: Optional[str] = None
    session_type: Optional[str] = None
    timeframe: Optional[str] = None


class AutoSetupStatus(BaseModel):
    """Status of auto-setup process."""
    status: str  # "ready", "checking", "training", "building_index", "error"
    model_exists: bool
    model_name: Optional[str] = None
    index_loaded: bool
    num_vectors: int
    num_sessions: int
    message: str
    # Training progress (if training)
    training_epoch: int = 0
    training_total_epochs: int = 0
    training_loss: float = 0.0
    # Training completion info (shown after training completes)
    last_training_completed: bool = False
    last_training_best_loss: float = 0.0
    last_training_epochs: int = 0


# --- Model Management ---

class ModelRenameRequest(BaseModel):
    """Request to rename a model."""
    working_directory: Optional[str] = None
    new_name: str


class ModelCopyRequest(BaseModel):
    """Request to copy/save-as a model."""
    working_directory: Optional[str] = None
    new_name: str


class ModelActionResponse(BaseModel):
    """Response from model management actions."""
    success: bool
    message: str


class DeleteResponse(BaseModel):
    """Response from bulk delete operations."""
    success: bool
    deleted_count: int
    message: str


# --- File Listing ---

class ParquetFileInfo(BaseModel):
    """Information about a parquet cache file."""
    pair: str
    timeframe: str
    file_name: str
    size_mb: float


class ModelSummary(BaseModel):
    """Summary of a trained model."""
    model_name: str
    latent_dim: int
    trained_epochs: int
    best_loss: float
    is_active: bool = False


class IndexSummary(BaseModel):
    """Summary of a vector index."""
    index_name: str
    num_vectors: int
    model_name: Optional[str] = None


class FileListResponse(BaseModel):
    """Comprehensive file listing response."""
    parquets: list[ParquetFileInfo]
    sessions_total: int
    sessions_by_pair: dict[str, int]
    sessions_by_session_type: dict[str, int]
    sessions_by_timeframe: dict[str, int]
    models: list[ModelSummary]
    indices: list[IndexSummary]
