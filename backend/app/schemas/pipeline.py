from pydantic import BaseModel
from typing import Optional
from app.config import TimeframeType


class ProcessRequest(BaseModel):
    working_directory: Optional[str] = None
    display_interval: TimeframeType = "M5"


class ProcessResponse(BaseModel):
    status: str
    instruments_processed: list[str]
    errors: list[str]
    message: str


class ProcessStatus(BaseModel):
    is_processing: bool
    progress: float
    current_file: Optional[str] = None
    message: str


class ProcessAllRequest(BaseModel):
    """Request for combined pipeline: CSV -> Parquet -> Sessions."""
    working_directory: Optional[str] = None
    pairs: Optional[list[str]] = None  # None = all detected pairs
    session_type: str = "ny"
    timeframe: TimeframeType = "M10"
    force_sessions: bool = False  # Force regenerate sessions
    max_sessions: Optional[int] = None  # Limit per pair (for testing)


class ProcessAllResponse(BaseModel):
    """Response from combined pipeline."""
    status: str
    parquets_created: int
    sessions_generated: int
    sessions_skipped: int
    total_time_seconds: float
    errors: list[str]
    message: str


class ProcessAllStatus(BaseModel):
    """Status of combined pipeline processing."""
    is_processing: bool
    stage: str  # "idle", "parquets", "sessions", "complete"
    progress: float
    current_pair: Optional[str] = None
    pairs_completed: int
    pairs_total: int
    message: str


class AvailablePairsResponse(BaseModel):
    """List of available pairs from CSVs."""
    pairs: list[str]
    csv_counts: dict[str, int]  # pair -> number of CSV files
