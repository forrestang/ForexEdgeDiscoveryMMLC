from fastapi import APIRouter, HTTPException, BackgroundTasks
from pathlib import Path
from typing import Optional
import time

from app.config import settings, TIMEFRAMES
from app.schemas.pipeline import (
    ProcessRequest,
    ProcessResponse,
    ProcessStatus,
    ProcessAllRequest,
    ProcessAllResponse,
    ProcessAllStatus,
    AvailablePairsResponse,
)
from app.core.csv_parser import scan_source_folder, load_and_stitch_pair_data
from app.core.timezone import est_to_utc
from app.core.upsampler import generate_all_timeframes
from app.core.cache_manager import save_to_cache, ensure_cache_directory, get_cached_instruments

router = APIRouter()

# Simple processing state (in production, use Redis or database)
_processing_state = {
    "is_processing": False,
    "progress": 0.0,
    "current_file": None,
    "message": "Idle",
}

# Combined pipeline state
_process_all_state = {
    "is_processing": False,
    "stage": "idle",
    "progress": 0.0,
    "current_pair": None,
    "pairs_completed": 0,
    "pairs_total": 0,
    "message": "Idle",
}


@router.post("/process", response_model=ProcessResponse)
async def process_data(request: ProcessRequest):
    """
    Process raw CSV files and create cached Parquet files.
    """
    global _processing_state

    if _processing_state["is_processing"]:
        raise HTTPException(status_code=409, detail="Processing already in progress")

    # Determine working directory
    working_dir = Path(request.working_directory) if request.working_directory else settings.default_working_directory
    data_path = working_dir / settings.data_folder_name
    cache_path = working_dir / settings.cache_folder_name

    if not data_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Data folder not found: {data_path}"
        )

    # Ensure cache directory exists
    ensure_cache_directory(cache_path)

    # Scan for files
    files_by_pair = scan_source_folder(data_path)

    if not files_by_pair:
        return ProcessResponse(
            status="warning",
            instruments_processed=[],
            errors=[],
            message="No MetaTrader CSV files found in data folder"
        )

    _processing_state["is_processing"] = True
    _processing_state["progress"] = 0.0
    _processing_state["message"] = "Starting processing..."

    instruments_processed = []
    errors = []
    total_pairs = len(files_by_pair)

    try:
        for idx, (pair, file_paths) in enumerate(files_by_pair.items()):
            try:
                _processing_state["current_file"] = pair
                _processing_state["message"] = f"Processing {pair}..."
                _processing_state["progress"] = idx / total_pairs

                # Load and stitch files
                df = load_and_stitch_pair_data(file_paths)

                if len(df) == 0:
                    errors.append(f"{pair}: No data loaded")
                    continue

                # Convert EST to UTC
                df = est_to_utc(df)

                # Generate all timeframes
                timeframe_dfs = generate_all_timeframes(df)

                # Save each timeframe to cache
                for tf, tf_df in timeframe_dfs.items():
                    save_to_cache(tf_df, pair, tf, cache_path)

                instruments_processed.append(pair)

            except Exception as e:
                errors.append(f"{pair}: {str(e)}")

        _processing_state["progress"] = 1.0
        _processing_state["message"] = "Processing complete"

    finally:
        _processing_state["is_processing"] = False
        _processing_state["current_file"] = None

    return ProcessResponse(
        status="success" if not errors else "partial",
        instruments_processed=instruments_processed,
        errors=errors,
        message=f"Processed {len(instruments_processed)} instruments"
    )


@router.get("/status", response_model=ProcessStatus)
async def get_status():
    """Get current processing status."""
    return ProcessStatus(
        is_processing=_processing_state["is_processing"],
        progress=_processing_state["progress"],
        current_file=_processing_state["current_file"],
        message=_processing_state["message"]
    )


# --- Combined Pipeline Endpoints ---

@router.get("/available-pairs", response_model=AvailablePairsResponse)
async def get_available_pairs(working_directory: Optional[str] = None):
    """
    Get list of available currency pairs from CSV files.

    Scans the data folder for MetaTrader CSV files and returns
    the detected pairs with their file counts.
    """
    working_dir = Path(working_directory) if working_directory else settings.default_working_directory
    data_path = working_dir / settings.data_folder_name

    if not data_path.exists():
        return AvailablePairsResponse(pairs=[], csv_counts={})

    files_by_pair = scan_source_folder(data_path)

    return AvailablePairsResponse(
        pairs=sorted(files_by_pair.keys()),
        csv_counts={pair: len(files) for pair, files in files_by_pair.items()}
    )


@router.post("/process-all", response_model=ProcessAllResponse)
async def process_all(request: ProcessAllRequest, background_tasks: BackgroundTasks):
    """
    Combined pipeline: CSV -> Parquet -> Sessions in one step.

    This endpoint:
    1. Scans for CSV files and processes them to Parquet (if needed)
    2. Generates session datasets from the Parquet files

    Use /process-all/status to monitor progress.
    """
    global _process_all_state

    if _process_all_state["is_processing"]:
        raise HTTPException(status_code=409, detail="Processing already in progress")

    working_dir = Path(request.working_directory) if request.working_directory else settings.default_working_directory
    data_path = working_dir / settings.data_folder_name
    cache_path = working_dir / settings.cache_folder_name

    if not data_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Data folder not found: {data_path}"
        )

    # Scan for available pairs
    files_by_pair = scan_source_folder(data_path)

    if not files_by_pair:
        return ProcessAllResponse(
            status="warning",
            parquets_created=0,
            parquets_skipped=0,
            sessions_generated=0,
            sessions_skipped=0,
            total_time_seconds=0.0,
            errors=[],
            message="No MetaTrader CSV files found in data folder"
        )

    # Filter to requested pairs
    if request.pairs:
        files_by_pair = {p: f for p, f in files_by_pair.items() if p in request.pairs}

    if not files_by_pair:
        return ProcessAllResponse(
            status="warning",
            parquets_created=0,
            parquets_skipped=0,
            sessions_generated=0,
            sessions_skipped=0,
            total_time_seconds=0.0,
            errors=[],
            message="No matching pairs found"
        )

    # Run in background
    background_tasks.add_task(
        _run_process_all,
        working_dir=working_dir,
        files_by_pair=files_by_pair,
        session_type=request.session_type,
        timeframe=request.timeframe,
        force_sessions=request.force_sessions,
        max_sessions=request.max_sessions,
    )

    return ProcessAllResponse(
        status="started",
        parquets_created=0,
        sessions_generated=0,
        sessions_skipped=0,
        total_time_seconds=0.0,
        errors=[],
        message=f"Processing started for {len(files_by_pair)} pairs"
    )


def _run_process_all(
    working_dir: Path,
    files_by_pair: dict,
    session_type: str,
    timeframe: str,
    force_sessions: bool,
    max_sessions: Optional[int],
):
    """Background task for combined pipeline processing."""
    global _process_all_state

    start_time = time.time()

    _process_all_state["is_processing"] = True
    _process_all_state["stage"] = "parquets"
    _process_all_state["progress"] = 0.0
    _process_all_state["pairs_completed"] = 0
    _process_all_state["pairs_total"] = len(files_by_pair)
    _process_all_state["message"] = "Starting..."

    cache_path = working_dir / settings.cache_folder_name
    ensure_cache_directory(cache_path)

    parquets_created = 0
    sessions_generated = 0
    sessions_skipped = 0
    errors = []

    pairs = list(files_by_pair.keys())

    try:
        # Stage 1: Generate parquets
        _process_all_state["stage"] = "parquets"
        _process_all_state["message"] = "Generating parquet files..."

        for idx, pair in enumerate(pairs):
            _process_all_state["current_pair"] = pair
            _process_all_state["progress"] = (idx / len(pairs)) * 0.5  # First half
            _process_all_state["message"] = f"Processing {pair} parquets..."

            try:
                # Load and stitch CSV files
                df = load_and_stitch_pair_data(files_by_pair[pair])

                if len(df) == 0:
                    errors.append(f"{pair}: No data loaded from CSVs")
                    continue

                # Convert EST to UTC
                df = est_to_utc(df)

                # Generate all timeframes
                timeframe_dfs = generate_all_timeframes(df)

                # Save each timeframe to cache (overwrites existing)
                for tf, tf_df in timeframe_dfs.items():
                    save_to_cache(tf_df, pair, tf, cache_path)

                parquets_created += 1

            except Exception as e:
                errors.append(f"{pair} parquet: {str(e)}")

        # Stage 2: Generate sessions
        _process_all_state["stage"] = "sessions"
        _process_all_state["message"] = "Generating session datasets..."

        from app.edge_finder.generator import generate_all_sessions

        for idx, pair in enumerate(pairs):
            _process_all_state["current_pair"] = pair
            _process_all_state["progress"] = 0.5 + (idx / len(pairs)) * 0.5  # Second half
            _process_all_state["pairs_completed"] = idx
            _process_all_state["message"] = f"Generating {pair} sessions..."

            try:
                stats = generate_all_sessions(
                    pair=pair,
                    timeframe=timeframe,
                    session_type=session_type,
                    working_directory=working_dir,
                    skip_existing=not force_sessions,
                )

                sessions_generated += stats["saved"]
                sessions_skipped += stats["skipped"]

                if stats["error_details"]:
                    for err in stats["error_details"][:3]:  # Limit errors
                        errors.append(f"{pair} session {err['date']}: {err['error']}")

            except Exception as e:
                errors.append(f"{pair} sessions: {str(e)}")

        _process_all_state["stage"] = "complete"
        _process_all_state["progress"] = 1.0
        _process_all_state["pairs_completed"] = len(pairs)

        elapsed = time.time() - start_time
        _process_all_state["message"] = f"Complete: {parquets_created} parquets, {sessions_generated} sessions in {elapsed:.1f}s"

    except Exception as e:
        _process_all_state["message"] = f"Error: {str(e)}"
        errors.append(str(e))

    finally:
        _process_all_state["is_processing"] = False
        _process_all_state["current_pair"] = None


@router.get("/process-all/status", response_model=ProcessAllStatus)
async def get_process_all_status():
    """Get status of combined pipeline processing."""
    return ProcessAllStatus(
        is_processing=_process_all_state["is_processing"],
        stage=_process_all_state["stage"],
        progress=_process_all_state["progress"],
        current_pair=_process_all_state["current_pair"],
        pairs_completed=_process_all_state["pairs_completed"],
        pairs_total=_process_all_state["pairs_total"],
        message=_process_all_state["message"],
    )
