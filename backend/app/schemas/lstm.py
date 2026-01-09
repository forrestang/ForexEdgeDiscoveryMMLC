from pydantic import BaseModel
from typing import Optional
from datetime import date


class DataFileInfo(BaseModel):
    """Info about a CSV data file."""
    name: str
    pair: Optional[str]
    year: Optional[int]
    type: str  # "data" or "test"
    size_mb: float


class DataFilesResponse(BaseModel):
    """Response for listing data files."""
    files: list[DataFileInfo]


class LSTMParquetInfo(BaseModel):
    """Info about an LSTM parquet file."""
    name: str
    pair: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    timeframe: Optional[str] = None
    adr_period: Optional[int] = None
    size_mb: float
    rows: int = 0


class LSTMParquetsResponse(BaseModel):
    """Response for listing LSTM parquets."""
    parquets: list[LSTMParquetInfo]


class CreateParquetRequest(BaseModel):
    """Request to create a date-range parquet."""
    pair: str
    start_date: date
    end_date: date
    working_directory: Optional[str] = None


class CreateParquetResponse(BaseModel):
    """Response from parquet creation."""
    status: str
    message: str
    filename: Optional[str]
    rows: int
    size_mb: float


class DeleteParquetResponse(BaseModel):
    """Response from parquet deletion."""
    status: str
    message: str


class DeleteParquetsRequest(BaseModel):
    """Request to delete multiple parquet files."""
    filenames: list[str]
    working_directory: Optional[str] = None


class DeleteParquetsResponse(BaseModel):
    """Response from batch parquet deletion."""
    status: str
    message: str
    deleted: list[str]
    errors: list[str]


class CreateFromFilesRequest(BaseModel):
    """Request to create parquets from selected files."""
    files: list[str]
    working_directory: Optional[str] = None
    adr_period: int = 20
    timeframes: list[str] = ["M5"]


class CreatedParquetInfo(BaseModel):
    """Info about a created parquet."""
    pair: str
    filename: str
    rows: int
    size_mb: float
    start_date: str
    end_date: str
    adr_period: int
    timeframe: str
    trimmed_days: int


class CreateFromFilesResponse(BaseModel):
    """Response from creating parquets from files."""
    status: str
    message: str
    created: list[CreatedParquetInfo]
    errors: list[str]


# ================================================================
# Bridge Schemas
# ================================================================

class BridgeRequest(BaseModel):
    """Request to bridge (enrich) a raw parquet file with MMLC state."""
    filename: str
    working_directory: Optional[str] = None


class BridgeResponse(BaseModel):
    """Response from bridge operation."""
    status: str
    message: str
    output_filename: Optional[str] = None
    rows: int = 0
    days_processed: int = 0
    processing_time_seconds: float = 0.0


class BridgedParquetInfo(BaseModel):
    """Info about a bridged parquet file."""
    name: str
    pair: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    timeframe: Optional[str] = None
    adr_period: Optional[int] = None
    size_mb: float
    rows: int = 0


class BridgedParquetsResponse(BaseModel):
    """Response for listing bridged parquets."""
    parquets: list[BridgedParquetInfo]


class RawParquetInfo(BaseModel):
    """Info about a raw parquet file available for bridging."""
    name: str
    pair: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    timeframe: Optional[str] = None
    adr_period: Optional[int] = None
    size_mb: float
    rows: int = 0


class RawParquetsForBridgeResponse(BaseModel):
    """Response for listing raw parquets available for bridging."""
    parquets: list[RawParquetInfo]


class BridgeFilesRequest(BaseModel):
    """Request to bridge multiple raw parquet files."""
    filenames: list[str]
    working_directory: Optional[str] = None


class BridgeFilesResponse(BaseModel):
    """Response from bridging multiple files."""
    status: str
    message: str
    results: list[BridgeResponse]
    errors: list[str]


