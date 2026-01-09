"""
Pydantic schemas for Transformer training API.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ================================================================
# Configuration Schemas
# ================================================================


class TransformerConfigRequest(BaseModel):
    """Request to start transformer training."""

    # Data settings
    target_session: str = Field(default="lon", description="Session prefix: asia, lon, ny, day")
    combine_sessions: Optional[str] = Field(default=None, description="Combined sessions: asia+lon, lon+ny")
    sequence_length: int = Field(default=64, ge=1, le=1000, description="Window size in bars")
    batch_size: int = Field(default=32, ge=1, le=256)

    # Model settings
    d_model: int = Field(default=128, ge=32, le=512)
    n_layers: int = Field(default=4, ge=1, le=12)
    n_heads: int = Field(default=4, ge=1, le=16)
    dropout_rate: float = Field(default=0.1, ge=0.0, le=0.5)

    # Training settings
    learning_rate: float = Field(default=1e-4, ge=1e-6, le=1e-2)
    num_epochs: int = Field(default=100, ge=1, le=1000)
    early_stopping_patience: int = Field(default=15, ge=1, le=100)

    # Model naming
    model_name: Optional[str] = Field(default=None, description="Custom model name (auto-generated if None)")
    save_to_models_folder: bool = Field(default=False, description="Save model to working_dir/models/")

    # Working directory
    working_directory: Optional[str] = None


class ConfigDefaultsRequest(BaseModel):
    """Request for configuration defaults based on session/timeframe."""

    session: str = Field(default="lon", description="Target session")
    timeframe: str = Field(default="M5", description="Bar timeframe")
    combine_sessions: Optional[str] = Field(default=None, description="Combined sessions")


class ConfigDefaultsResponse(BaseModel):
    """Response with configuration defaults."""

    sequence_length: int
    session_hours: int
    timeframe_minutes: int
    available_sessions: list[str]
    available_timeframes: list[str]


# ================================================================
# Training Status Schemas
# ================================================================


class TrainingStatusResponse(BaseModel):
    """Response with current training status."""

    status: str  # idle, training, stopping, error
    current_epoch: int = 0
    total_epochs: int = 0
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    best_loss: Optional[float] = None
    epochs_without_improvement: int = 0
    learning_rate: Optional[float] = None
    model_name: Optional[str] = None
    message: Optional[str] = None

    # Elapsed time
    elapsed_seconds: float = 0.0

    # Additional metrics
    directional_accuracy: Optional[float] = None  # 0-100%
    r_squared: Optional[float] = None  # 0.0 to 1.0
    max_error: Optional[float] = None  # Same units as target
    grad_norm: Optional[float] = None  # Running average


class StartTrainingResponse(BaseModel):
    """Response from starting training."""

    status: str  # started, error
    message: str
    model_name: Optional[str] = None


class StopTrainingResponse(BaseModel):
    """Response from stopping training."""

    status: str  # stopped, not_running, error
    message: str


# ================================================================
# Model Management Schemas
# ================================================================


class TransformerModelInfo(BaseModel):
    """Info about a saved transformer model."""

    name: str
    best_loss: Optional[float] = None
    epochs_trained: int = 0
    target_session: Optional[str] = None
    sequence_length: Optional[int] = None
    d_model: Optional[int] = None
    n_layers: Optional[int] = None
    created_at: Optional[str] = None


class TransformerModelsResponse(BaseModel):
    """Response for listing transformer models."""

    models: list[TransformerModelInfo]


class DeleteModelRequest(BaseModel):
    """Request to delete a model."""

    model_name: str
    working_directory: Optional[str] = None


class DeleteModelResponse(BaseModel):
    """Response from model deletion."""

    status: str  # deleted, not_found, error
    message: str


# ================================================================
# Validation Schemas
# ================================================================


class ValidationGenerateRequest(BaseModel):
    """Request to generate validation test data."""

    test_type: str = Field(description="Test type: sanity, memory, or all")
    n_sessions: int = Field(default=500, ge=100, le=5000)
    seq_len: int = Field(default=64, ge=16, le=256)
    working_directory: Optional[str] = None


class ValidationGenerateResponse(BaseModel):
    """Response from generating validation data."""

    status: str  # success, error
    message: str
    files_created: list[str] = []


class ValidationRunRequest(BaseModel):
    """Request to run a validation test."""

    test_type: str = Field(description="Test type: sanity or memory")
    num_epochs: int = Field(default=20, ge=5, le=100)
    working_directory: Optional[str] = None


class ValidationRunResponse(BaseModel):
    """Response from starting a validation run."""

    status: str  # started, error
    message: str


class ValidationStatusResponse(BaseModel):
    """Response with validation run status."""

    status: str  # idle, running, complete, error
    test_type: Optional[str] = None
    passed: Optional[bool] = None
    best_loss: Optional[float] = None
    threshold: Optional[float] = None
    current_epoch: int = 0
    total_epochs: int = 0
    message: Optional[str] = None


# ================================================================
# Parquet Viewer Schemas
# ================================================================


class ParquetFileInfo(BaseModel):
    """Info about a parquet file."""

    name: str
    path: str
    rows: int
    size_mb: float
    has_mmlc_columns: bool = False


class ParquetFilesResponse(BaseModel):
    """Response listing available parquet files."""

    files: list[ParquetFileInfo]


class CandleData(BaseModel):
    """Single candle data."""

    timestamp: str
    open: float
    high: float
    low: float
    close: float


class StateData(BaseModel):
    """MMLC state data for a single bar."""

    level: Optional[int] = None
    event: Optional[str] = None
    dir: Optional[str] = None
    out_max_up: Optional[float] = None
    out_max_down: Optional[float] = None


class ParquetDataResponse(BaseModel):
    """Response with parquet data for visualization."""

    candles: list[CandleData]
    states: list[StateData]
    total_rows: int
    start_idx: int
    session: str
    date: Optional[str] = None  # Date filter if applied


class ParquetDatesResponse(BaseModel):
    """Response listing available dates in a parquet file."""

    filename: str
    dates: list[str]
    total_dates: int


# ================================================================
# Training Queue Schemas
# ================================================================


class TrainingCardConfig(BaseModel):
    """Configuration for a single training card in the queue."""

    card_id: str
    model_name: str
    parquet_file: str  # Specific parquet file to train on
    session_option: str  # asia, lon, ny, day, asia_lon, lon_ny
    sequence_length: int = 64
    batch_size: int = 32
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    dropout_rate: float = 0.1
    learning_rate: float = 1e-4
    num_epochs: int = 100
    early_stopping_patience: int = 15
    save_to_models_folder: bool = True


class TrainingCardStatus(BaseModel):
    """Status of a single training card."""

    card_id: str
    status: str  # pending, training, completed, error
    model_name: str
    current_epoch: int = 0
    total_epochs: int = 0
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    best_loss: Optional[float] = None
    elapsed_seconds: float = 0.0
    directional_accuracy: Optional[float] = None
    r_squared: Optional[float] = None
    max_error: Optional[float] = None
    grad_norm: Optional[float] = None
    error_message: Optional[str] = None

    # Final report metrics (populated when completed)
    final_directional_accuracy: Optional[float] = None
    final_r_squared: Optional[float] = None
    final_max_error: Optional[float] = None


class QueueStatusResponse(BaseModel):
    """Response for queue status."""

    queue_running: bool
    current_card_id: Optional[str] = None
    cards: list[TrainingCardStatus]


class AddToQueueRequest(BaseModel):
    """Request to add a training card to the queue."""

    config: TrainingCardConfig
    working_directory: Optional[str] = None


class AddToQueueResponse(BaseModel):
    """Response from adding to queue."""

    status: str  # added, error
    card_id: str
    message: Optional[str] = None


class RemoveFromQueueRequest(BaseModel):
    """Request to remove a card from the queue."""

    card_id: str


class RemoveFromQueueResponse(BaseModel):
    """Response from removing from queue."""

    status: str  # removed, not_found, error
    card_id: Optional[str] = None
    message: Optional[str] = None


class StartQueueRequest(BaseModel):
    """Request to start processing the training queue."""

    working_directory: Optional[str] = None


class StartQueueResponse(BaseModel):
    """Response from starting the queue."""

    status: str  # started, error
    message: Optional[str] = None


class StopQueueResponse(BaseModel):
    """Response from stopping the queue."""

    status: str  # stopping, not_running
    message: Optional[str] = None
