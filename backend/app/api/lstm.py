from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from app.schemas.lstm import (
    DataFilesResponse,
    DataFileInfo,
    LSTMParquetsResponse,
    LSTMParquetInfo,
    CreateParquetRequest,
    CreateParquetResponse,
    DeleteParquetResponse,
    DeleteParquetsRequest,
    DeleteParquetsResponse,
    CreateFromFilesRequest,
    CreateFromFilesResponse,
    CreatedParquetInfo,
    BridgeRequest,
    BridgeResponse,
    BridgedParquetsResponse,
    BridgedParquetInfo,
    RawParquetsForBridgeResponse,
    RawParquetInfo,
    BridgeFilesRequest,
    BridgeFilesResponse,
)
from app.lstm.data_processor import (
    scan_data_files,
    list_lstm_parquets,
    create_date_range_parquet,
    delete_lstm_parquet,
    delete_lstm_parquets_batch,
    create_parquets_from_files,
    get_data_path,
)

router = APIRouter()


@router.get("/data-files", response_model=DataFilesResponse)
async def get_data_files(
    working_directory: Optional[str] = Query(None, description="Working directory path")
):
    """
    List CSV files in the data folder.
    Returns file metadata including pair name and year extracted from filename.
    """
    files = scan_data_files(working_directory)
    return DataFilesResponse(
        files=[DataFileInfo(**f) for f in files]
    )


@router.get("/parquets", response_model=LSTMParquetsResponse)
async def get_lstm_parquets(
    working_directory: Optional[str] = Query(None, description="Working directory path")
):
    """
    List existing LSTM parquet files in lstm folder.
    """
    parquets = list_lstm_parquets(working_directory)
    return LSTMParquetsResponse(
        parquets=[LSTMParquetInfo(**p) for p in parquets]
    )


@router.post("/create-parquet", response_model=CreateParquetResponse)
async def create_parquet(request: CreateParquetRequest):
    """
    Create a date-range parquet file for a specific pair.
    Loads all CSV files for the pair, filters to the date range, and saves to lstm folder.
    """
    result = create_date_range_parquet(
        pair=request.pair,
        start_date=request.start_date,
        end_date=request.end_date,
        working_dir=request.working_directory,
    )

    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])

    return CreateParquetResponse(**result)


@router.post("/create-from-files", response_model=CreateFromFilesResponse)
async def create_from_files(request: CreateFromFilesRequest):
    """
    Create parquet files from selected CSV files.
    Groups files by pair, calculates ADR, upsamples to selected timeframes,
    and creates one parquet per pair per timeframe.
    """
    result = create_parquets_from_files(
        selected_files=request.files,
        working_dir=request.working_directory,
        adr_period=request.adr_period,
        timeframes=request.timeframes,
    )

    return CreateFromFilesResponse(
        status=result["status"],
        message=result["message"],
        created=[CreatedParquetInfo(**c) for c in result["created"]],
        errors=result["errors"],
    )


@router.delete("/parquet/{filename}", response_model=DeleteParquetResponse)
async def delete_parquet(
    filename: str,
    working_directory: Optional[str] = Query(None, description="Working directory path")
):
    """
    Delete an LSTM parquet file.
    """
    result = delete_lstm_parquet(filename, working_directory)

    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result["message"])

    return DeleteParquetResponse(**result)


@router.post("/parquets/delete-batch", response_model=DeleteParquetsResponse)
async def delete_parquets_batch(request: DeleteParquetsRequest):
    """
    Delete multiple LSTM parquet files.
    """
    result = delete_lstm_parquets_batch(
        filenames=request.filenames,
        working_dir=request.working_directory,
    )

    return DeleteParquetsResponse(
        status=result["status"],
        message=result["message"],
        deleted=result["deleted"],
        errors=result["errors"],
    )


# ================================================================
# Bridge Endpoints
# ================================================================

@router.get("/bridge/raw-parquets", response_model=RawParquetsForBridgeResponse)
async def get_raw_parquets_for_bridge(
    working_directory: Optional[str] = Query(None, description="Working directory path")
):
    """
    List raw parquet files available for bridging.
    """
    from app.lstm.bridge import LSTMBridge

    bridge = LSTMBridge(working_directory)
    parquets = bridge.list_raw_parquets()
    return RawParquetsForBridgeResponse(
        parquets=[RawParquetInfo(**p) for p in parquets]
    )


@router.get("/bridge/bridged-parquets", response_model=BridgedParquetsResponse)
async def get_bridged_parquets(
    working_directory: Optional[str] = Query(None, description="Working directory path")
):
    """
    List existing bridged parquet files.
    """
    from app.lstm.bridge import LSTMBridge

    bridge = LSTMBridge(working_directory)
    parquets = bridge.list_bridged_parquets()
    return BridgedParquetsResponse(
        parquets=[BridgedParquetInfo(**p) for p in parquets]
    )


@router.post("/bridge", response_model=BridgeResponse)
async def bridge_parquet(request: BridgeRequest):
    """
    Enrich a raw parquet file with MMLC state and outcomes.
    Creates output in lstm/bridged/ folder.

    This runs the MMLC algorithm on each bar for 4 sessions (Asia, London, NY, Full Day)
    and adds 28 new columns containing level, direction, event, and outcome data.
    """
    from app.lstm.bridge import LSTMBridge

    bridge = LSTMBridge(request.working_directory)
    result = bridge.enrich_file_optimized(request.filename)

    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])

    return BridgeResponse(**result)


@router.delete("/bridge/bridged/{filename}", response_model=DeleteParquetResponse)
async def delete_bridged_parquet(
    filename: str,
    working_directory: Optional[str] = Query(None, description="Working directory path")
):
    """
    Delete a bridged parquet file.
    """
    from app.lstm.bridge import LSTMBridge

    bridge = LSTMBridge(working_directory)
    result = bridge.delete_bridged(filename)

    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result["message"])

    return DeleteParquetResponse(**result)


@router.post("/bridge/batch", response_model=BridgeFilesResponse)
async def bridge_multiple_parquets(request: BridgeFilesRequest):
    """
    Enrich multiple raw parquet files with MMLC state and outcomes.
    Creates output files in lstm/bridged/ folder.
    """
    from app.lstm.bridge import LSTMBridge

    bridge = LSTMBridge(request.working_directory)
    result = bridge.enrich_multiple_files(request.filenames)

    return BridgeFilesResponse(
        status=result["status"],
        message=result["message"],
        results=[BridgeResponse(**r) for r in result["results"]],
        errors=result["errors"],
    )


