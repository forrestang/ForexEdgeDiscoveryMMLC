"""
Transformer Training API endpoints.

Provides REST API for transformer model training, status monitoring,
and model management.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pathlib import Path
from typing import Optional
import threading
import json
import time

from app.config import settings
from app.schemas.transformer import (
    TransformerConfigRequest,
    ConfigDefaultsRequest,
    ConfigDefaultsResponse,
    TrainingStatusResponse,
    StartTrainingResponse,
    StopTrainingResponse,
    TransformerModelInfo,
    TransformerModelsResponse,
    DeleteModelRequest,
    DeleteModelResponse,
    ValidationGenerateRequest,
    ValidationGenerateResponse,
    ValidationRunRequest,
    ValidationRunResponse,
    ValidationStatusResponse,
    ParquetFileInfo,
    ParquetFilesResponse,
    CandleData,
    StateData,
    ParquetDataResponse,
    ParquetDatesResponse,
    # Queue schemas
    TrainingCardConfig,
    TrainingCardStatus,
    QueueStatusResponse,
    AddToQueueRequest,
    AddToQueueResponse,
    RemoveFromQueueRequest,
    RemoveFromQueueResponse,
    StartQueueRequest,
    StartQueueResponse,
    StopQueueResponse,
)
from app.transformer.config import (
    get_default_sequence_length,
    SESSION_HOURS,
    TIMEFRAME_MINUTES,
)

router = APIRouter()

# --- Global Training State ---

_training_state = {
    "status": "idle",  # idle, training, stopping, error
    "current_epoch": 0,
    "total_epochs": 0,
    "train_loss": None,
    "val_loss": None,
    "best_loss": None,
    "epochs_without_improvement": 0,
    "learning_rate": None,
    "model_name": None,
    "message": None,
    "stop_requested": False,
    # Elapsed time tracking
    "start_time": None,  # time.time() when training started
    # Additional metrics
    "directional_accuracy": None,
    "r_squared": None,
    "max_error": None,
    "grad_norm": None,
}

_training_lock = threading.Lock()

# --- Queue State ---

_queue_state = {
    "running": False,
    "cards": [],  # List of {config: dict, status: dict, working_directory: str}
    "current_card_id": None,
}

_queue_lock = threading.Lock()


def _get_working_dir(working_directory: Optional[str]) -> Path:
    """Get working directory, defaulting to settings."""
    if working_directory:
        return Path(working_directory)
    return settings.default_working_directory


# --- Configuration Endpoints ---


@router.get("/config-defaults", response_model=ConfigDefaultsResponse)
async def get_config_defaults(
    session: str = "lon",
    timeframe: str = "M5",
    combine_sessions: Optional[str] = None,
):
    """
    Get default configuration values based on session and timeframe.

    The sequence_length is automatically calculated based on session hours
    and timeframe minutes.
    """
    seq_length = get_default_sequence_length(session, timeframe, combine_sessions)

    if combine_sessions:
        prefixes = combine_sessions.split("+")
        total_hours = sum(SESSION_HOURS.get(p, 9) for p in prefixes)
    else:
        total_hours = SESSION_HOURS.get(session, 9)

    return ConfigDefaultsResponse(
        sequence_length=seq_length,
        session_hours=total_hours,
        timeframe_minutes=TIMEFRAME_MINUTES.get(timeframe, 5),
        available_sessions=list(SESSION_HOURS.keys()),
        available_timeframes=list(TIMEFRAME_MINUTES.keys()),
    )


# --- Training Endpoints ---


@router.get("/status", response_model=TrainingStatusResponse)
async def get_training_status():
    """Get current training status."""
    with _training_lock:
        # Calculate elapsed time
        elapsed_seconds = 0.0
        if _training_state["start_time"] and _training_state["status"] == "training":
            elapsed_seconds = time.time() - _training_state["start_time"]

        return TrainingStatusResponse(
            status=_training_state["status"],
            current_epoch=_training_state["current_epoch"],
            total_epochs=_training_state["total_epochs"],
            train_loss=_training_state["train_loss"],
            val_loss=_training_state["val_loss"],
            best_loss=_training_state["best_loss"],
            epochs_without_improvement=_training_state["epochs_without_improvement"],
            learning_rate=_training_state["learning_rate"],
            model_name=_training_state["model_name"],
            message=_training_state["message"],
            elapsed_seconds=elapsed_seconds,
            directional_accuracy=_training_state["directional_accuracy"],
            r_squared=_training_state["r_squared"],
            max_error=_training_state["max_error"],
            grad_norm=_training_state["grad_norm"],
        )


@router.post("/start-training", response_model=StartTrainingResponse)
async def start_training(
    request: TransformerConfigRequest,
    background_tasks: BackgroundTasks,
):
    """
    Start transformer model training.

    Training runs in the background. Use /status to monitor progress.
    """
    global _training_state

    with _training_lock:
        if _training_state["status"] == "training":
            raise HTTPException(status_code=409, detail="Training already in progress")

    working_dir = _get_working_dir(request.working_directory)

    # Generate model name if not provided
    model_name = request.model_name
    if not model_name:
        from app.transformer.trainer import generate_model_name
        # Try to extract ADR period from available parquets
        model_name = generate_model_name(
            adr_period=20,  # Default
            bar_interval="M5",  # Default
            session=request.target_session,
        )

    # Start training in background
    background_tasks.add_task(
        _run_training,
        working_dir=working_dir,
        model_name=model_name,
        request=request,
    )

    return StartTrainingResponse(
        status="started",
        message=f"Training started for model '{model_name}'",
        model_name=model_name,
    )


def _run_training(
    working_dir: Path,
    model_name: str,
    request: TransformerConfigRequest,
):
    """Background task for transformer training."""
    global _training_state

    try:
        from app.transformer.trainer import TransformerTrainer
        from app.transformer.config import TransformerConfig

        with _training_lock:
            _training_state["status"] = "training"
            _training_state["model_name"] = model_name
            _training_state["total_epochs"] = request.num_epochs
            _training_state["current_epoch"] = 0
            _training_state["stop_requested"] = False
            _training_state["message"] = "Initializing trainer..."
            _training_state["best_loss"] = None
            _training_state["train_loss"] = None
            _training_state["val_loss"] = None
            # Reset elapsed time and new metrics
            _training_state["start_time"] = time.time()
            _training_state["directional_accuracy"] = None
            _training_state["r_squared"] = None
            _training_state["max_error"] = None
            _training_state["grad_norm"] = None

        # Create config
        config = TransformerConfig(
            target_session=request.target_session,
            combine_sessions=request.combine_sessions,
            sequence_length=request.sequence_length,
            batch_size=request.batch_size,
            d_model=request.d_model,
            n_layers=request.n_layers,
            n_heads=request.n_heads,
            dropout_rate=request.dropout_rate,
            learning_rate=request.learning_rate,
            num_epochs=request.num_epochs,
            early_stopping_patience=request.early_stopping_patience,
        )

        trainer = TransformerTrainer(
            config=config,
            working_directory=working_dir,
            model_name=model_name,
        )

        # Setup data loaders and model
        with _training_lock:
            _training_state["message"] = "Loading parquet data..."

        trainer.setup()

        # Set up epoch callback for progress updates
        def on_epoch_end(epoch, metrics):
            with _training_lock:
                _training_state["current_epoch"] = epoch + 1
                _training_state["train_loss"] = metrics.get("loss", 0.0)
                _training_state["val_loss"] = metrics.get("val_loss", 0.0)
                _training_state["learning_rate"] = trainer.optimizer.param_groups[0]["lr"]
                _training_state["message"] = f"Epoch {epoch + 1}/{request.num_epochs}"

                # New metrics
                _training_state["grad_norm"] = metrics.get("grad_norm")
                _training_state["directional_accuracy"] = metrics.get("directional_accuracy")
                _training_state["r_squared"] = metrics.get("r_squared")
                _training_state["max_error"] = metrics.get("max_error")

                if trainer.state:
                    _training_state["best_loss"] = trainer.state.best_loss
                    _training_state["epochs_without_improvement"] = trainer.state.epochs_without_improvement

                # Check for stop request
                if _training_state["stop_requested"]:
                    raise StopIteration("Training stopped by user")

        trainer.on_epoch_end = on_epoch_end

        # Run training
        with _training_lock:
            _training_state["message"] = "Training..."

        trainer.train()

        # Export model if requested
        if request.save_to_models_folder:
            trainer.export_model(save_to_models_folder=True)

        with _training_lock:
            _training_state["status"] = "idle"
            _training_state["message"] = "Training complete"
            _training_state["start_time"] = None

    except StopIteration:
        with _training_lock:
            _training_state["status"] = "idle"
            _training_state["message"] = "Training stopped by user"
            _training_state["start_time"] = None

    except Exception as e:
        with _training_lock:
            _training_state["status"] = "error"
            _training_state["message"] = f"Error: {str(e)}"
            _training_state["start_time"] = None


@router.post("/stop-training", response_model=StopTrainingResponse)
async def stop_training():
    """Request to stop the current training."""
    global _training_state

    with _training_lock:
        if _training_state["status"] != "training":
            return StopTrainingResponse(
                status="not_running",
                message="No training is currently running",
            )

        _training_state["stop_requested"] = True
        _training_state["status"] = "stopping"
        _training_state["message"] = "Stopping training..."

    return StopTrainingResponse(
        status="stopping",
        message="Stop request sent. Training will stop after current epoch.",
    )


# --- Model Management Endpoints ---


@router.get("/models", response_model=TransformerModelsResponse)
async def list_models(working_directory: Optional[str] = None):
    """List all saved transformer models."""
    working_dir = _get_working_dir(working_directory)

    try:
        from app.transformer.storage import (
            get_transformer_models_path,
            list_transformer_models,
        )

        model_names = list_transformer_models(working_dir)
        models_path = get_transformer_models_path(working_dir)

        models = []
        for name in model_names:
            info = TransformerModelInfo(name=name)

            # Try to load config for additional details
            config_path = models_path / f"{name}_config.json"
            if config_path.exists():
                try:
                    with open(config_path, "r") as f:
                        config_data = json.load(f)
                    info.target_session = config_data.get("target_session")
                    info.sequence_length = config_data.get("sequence_length")
                    info.d_model = config_data.get("d_model")
                    info.n_layers = config_data.get("n_layers")
                except Exception:
                    pass

            # Try to load state for training details
            state_path = models_path / f"{name}_state.json"
            if state_path.exists():
                try:
                    with open(state_path, "r") as f:
                        state_data = json.load(f)
                    info.best_loss = state_data.get("best_loss")
                    info.epochs_trained = state_data.get("epoch", 0)
                except Exception:
                    pass

            models.append(info)

        return TransformerModelsResponse(models=models)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/delete-model", response_model=DeleteModelResponse)
async def delete_model(request: DeleteModelRequest):
    """Delete a saved model and its associated files."""
    working_dir = _get_working_dir(request.working_directory)

    try:
        from app.transformer.storage import get_transformer_models_path

        models_path = get_transformer_models_path(working_dir)
        model_name = request.model_name

        # Find and delete all files for this model
        patterns = [
            f"{model_name}_checkpoint.pt",
            f"{model_name}_best.pt",
            f"{model_name}_config.json",
            f"{model_name}_state.json",
            f"{model_name}.pt",
        ]

        deleted = False
        for pattern in patterns:
            file_path = models_path / pattern
            if file_path.exists():
                file_path.unlink()
                deleted = True

        if deleted:
            return DeleteModelResponse(
                status="deleted",
                message=f"Model '{model_name}' deleted successfully",
            )
        else:
            return DeleteModelResponse(
                status="not_found",
                message=f"Model '{model_name}' not found",
            )

    except Exception as e:
        return DeleteModelResponse(
            status="error",
            message=str(e),
        )


# --- Validation State ---

_validation_state = {
    "status": "idle",  # idle, running, complete, error
    "test_type": None,
    "passed": None,
    "best_loss": None,
    "threshold": None,
    "current_epoch": 0,
    "total_epochs": 0,
    "message": None,
}

_validation_lock = threading.Lock()


# --- Validation Endpoints ---


@router.post("/validation/generate", response_model=ValidationGenerateResponse)
async def generate_validation_data(request: ValidationGenerateRequest):
    """Generate synthetic validation test data."""
    working_dir = _get_working_dir(request.working_directory)

    try:
        from app.transformer.validation import generate_sanity_check, generate_memory_test

        files_created = []

        if request.test_type in ("sanity", "all"):
            path = generate_sanity_check(
                working_dir,
                n_sessions=request.n_sessions,
                seq_len=request.seq_len,
            )
            files_created.append(path.name)

        if request.test_type in ("memory", "all"):
            path = generate_memory_test(
                working_dir,
                n_sessions=request.n_sessions,
                seq_len=request.seq_len,
            )
            files_created.append(path.name)

        return ValidationGenerateResponse(
            status="success",
            message=f"Generated {len(files_created)} validation file(s)",
            files_created=files_created,
        )

    except Exception as e:
        return ValidationGenerateResponse(
            status="error",
            message=str(e),
            files_created=[],
        )


@router.post("/validation/run", response_model=ValidationRunResponse)
async def run_validation(
    request: ValidationRunRequest,
    background_tasks: BackgroundTasks,
):
    """Run a validation test in the background."""
    global _validation_state

    with _validation_lock:
        if _validation_state["status"] == "running":
            raise HTTPException(status_code=409, detail="Validation already running")

    working_dir = _get_working_dir(request.working_directory)

    # Start validation in background
    background_tasks.add_task(
        _run_validation_task,
        working_dir=working_dir,
        test_type=request.test_type,
        num_epochs=request.num_epochs,
    )

    return ValidationRunResponse(
        status="started",
        message=f"Validation '{request.test_type}' started",
    )


def _run_validation_task(working_dir: Path, test_type: str, num_epochs: int):
    """Background task for validation."""
    global _validation_state

    try:
        from app.transformer.validation import generate_sanity_check, generate_memory_test

        # Set thresholds
        thresholds = {"sanity": 0.001, "memory": 0.01}
        threshold = thresholds.get(test_type, 0.001)

        with _validation_lock:
            _validation_state["status"] = "running"
            _validation_state["test_type"] = test_type
            _validation_state["passed"] = None
            _validation_state["best_loss"] = None
            _validation_state["threshold"] = threshold
            _validation_state["current_epoch"] = 0
            _validation_state["total_epochs"] = num_epochs
            _validation_state["message"] = "Generating test data..."

        # Generate test data first
        if test_type == "sanity":
            parquet_path = generate_sanity_check(working_dir, n_sessions=500, seq_len=64)
        else:
            parquet_path = generate_memory_test(working_dir, n_sessions=500, seq_len=64)

        with _validation_lock:
            _validation_state["message"] = "Initializing trainer..."

        # Import and run validation
        from app.transformer.config import TransformerConfig
        from app.transformer.trainer import TransformerTrainer

        config = TransformerConfig(
            target_session="lon",
            sequence_length=64,
            batch_size=64,
            d_model=64,
            n_layers=2,
            n_heads=2,
            num_epochs=num_epochs,
            learning_rate=1e-3,
            early_stopping_patience=10,
            save_every=100,
        )

        trainer = TransformerTrainer(
            config=config,
            working_directory=working_dir,
            model_name=f"validation_{test_type}",
        )

        # Setup with specific parquet (duplicate for train/val)
        trainer.setup(parquet_files=[parquet_path, parquet_path], val_ratio=0.5)

        # Set up progress callback
        def on_epoch_end(epoch, metrics):
            with _validation_lock:
                _validation_state["current_epoch"] = epoch + 1
                _validation_state["message"] = f"Epoch {epoch + 1}/{num_epochs}"
                if trainer.state:
                    _validation_state["best_loss"] = trainer.state.best_loss

        trainer.on_epoch_end = on_epoch_end

        with _validation_lock:
            _validation_state["message"] = "Training..."

        # Train
        state = trainer.train()

        # Determine pass/fail
        passed = state.best_loss < threshold

        with _validation_lock:
            _validation_state["status"] = "complete"
            _validation_state["passed"] = passed
            _validation_state["best_loss"] = state.best_loss
            _validation_state["message"] = "PASSED" if passed else "FAILED"

    except Exception as e:
        with _validation_lock:
            _validation_state["status"] = "error"
            _validation_state["message"] = str(e)


@router.get("/validation/status", response_model=ValidationStatusResponse)
async def get_validation_status():
    """Get current validation run status."""
    with _validation_lock:
        return ValidationStatusResponse(
            status=_validation_state["status"],
            test_type=_validation_state["test_type"],
            passed=_validation_state["passed"],
            best_loss=_validation_state["best_loss"],
            threshold=_validation_state["threshold"],
            current_epoch=_validation_state["current_epoch"],
            total_epochs=_validation_state["total_epochs"],
            message=_validation_state["message"],
        )


# --- Parquet Viewer Endpoints ---


@router.get("/parquet-files", response_model=ParquetFilesResponse)
async def list_parquet_files(working_directory: Optional[str] = None):
    """List all available parquet files for viewing."""
    working_dir = _get_working_dir(working_directory)

    try:
        import polars as pl

        files = []

        # Check lstm/ folder
        lstm_path = working_dir / "lstm"
        if lstm_path.exists():
            for pq in lstm_path.glob("*.parquet"):
                try:
                    df = pl.scan_parquet(pq)
                    schema = df.collect_schema()
                    rows = df.select(pl.len()).collect().item()
                    size_mb = pq.stat().st_size / (1024 * 1024)

                    has_mmlc = any(col.endswith("_state_level") for col in schema.names())

                    files.append(ParquetFileInfo(
                        name=pq.name,
                        path=f"lstm/{pq.name}",
                        rows=rows,
                        size_mb=round(size_mb, 2),
                        has_mmlc_columns=has_mmlc,
                    ))
                except Exception:
                    pass

        # Check lstm/bridged/ folder
        bridged_path = working_dir / "lstm" / "bridged"
        if bridged_path.exists():
            for pq in bridged_path.glob("*.parquet"):
                try:
                    df = pl.scan_parquet(pq)
                    schema = df.collect_schema()
                    rows = df.select(pl.len()).collect().item()
                    size_mb = pq.stat().st_size / (1024 * 1024)

                    has_mmlc = any(col.endswith("_state_level") for col in schema.names())

                    files.append(ParquetFileInfo(
                        name=pq.name,
                        path=f"lstm/bridged/{pq.name}",
                        rows=rows,
                        size_mb=round(size_mb, 2),
                        has_mmlc_columns=has_mmlc,
                    ))
                except Exception:
                    pass

        return ParquetFilesResponse(files=files)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/parquet-dates", response_model=ParquetDatesResponse)
async def get_parquet_dates(
    filename: str,
    working_directory: Optional[str] = None,
):
    """Get list of unique dates available in a parquet file."""
    working_dir = _get_working_dir(working_directory)

    try:
        import polars as pl
        from app.core.session_filter import get_available_dates

        file_path = working_dir / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")

        df = pl.read_parquet(file_path)
        dates = get_available_dates(df)

        return ParquetDatesResponse(
            filename=filename,
            dates=[str(d) for d in sorted(dates)],
            total_dates=len(dates),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/parquet-data", response_model=ParquetDataResponse)
async def get_parquet_data(
    filename: str,
    session: str = "lon",
    start_idx: int = 0,
    limit: int = 500,
    date: Optional[str] = None,
    working_directory: Optional[str] = None,
):
    """Load parquet data for visualization with optional date filtering.

    Note: Session parameter only controls which MMLC state columns to display
    (e.g., lon_state_level, ny_state_level). It does NOT filter price bars by
    session time, because parquet files contain pre-upsampled data (M10, M5, etc.)
    and session time filtering would corrupt the data.
    """
    working_dir = _get_working_dir(working_directory)

    try:
        import polars as pl
        from datetime import date as date_type
        from app.core.session_filter import filter_by_date

        # Resolve file path
        file_path = working_dir / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")

        # Load parquet
        df = pl.read_parquet(file_path)

        # Apply date filter if provided
        if date:
            target_date = date_type.fromisoformat(date)
            df = filter_by_date(df, target_date)

        total_rows = len(df)

        # Handle empty result
        if total_rows == 0:
            return ParquetDataResponse(
                candles=[],
                states=[],
                total_rows=0,
                start_idx=0,
                session=session,
                date=date,
            )

        # Clamp indices
        start_idx = max(0, min(start_idx, total_rows - 1))
        end_idx = min(start_idx + limit, total_rows)

        # Slice data
        slice_df = df.slice(start_idx, end_idx - start_idx)

        # Extract candle data
        candles = []
        for row in slice_df.iter_rows(named=True):
            candles.append(CandleData(
                timestamp=str(row.get("timestamp", "")),
                open=float(row.get("open", 0)),
                high=float(row.get("high", 0)),
                low=float(row.get("low", 0)),
                close=float(row.get("close", 0)),
            ))

        # Extract MMLC state data if available
        states = []
        level_col = f"{session}_state_level"
        event_col = f"{session}_state_event"
        dir_col = f"{session}_state_dir"
        out_up_col = f"{session}_out_max_up"
        out_down_col = f"{session}_out_max_down"

        for row in slice_df.iter_rows(named=True):
            states.append(StateData(
                level=row.get(level_col),
                event=row.get(event_col),
                dir=row.get(dir_col),
                out_max_up=row.get(out_up_col),
                out_max_down=row.get(out_down_col),
            ))

        return ParquetDataResponse(
            candles=candles,
            states=states,
            total_rows=total_rows,
            start_idx=start_idx,
            session=session,
            date=date,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Queue Endpoints ---


def _parse_session_option(session_option: str) -> tuple[str, Optional[str]]:
    """
    Convert unified session option to target_session and combine_sessions.

    Options:
    - "asia" -> ("asia", None)
    - "lon" -> ("lon", None)
    - "ny" -> ("ny", None)
    - "day" -> ("day", None)
    - "asia_lon" -> ("asia", "asia+lon")
    - "lon_ny" -> ("lon", "lon+ny")
    """
    mapping = {
        "asia": ("asia", None),
        "lon": ("lon", None),
        "ny": ("ny", None),
        "day": ("day", None),
        "asia_lon": ("asia", "asia+lon"),
        "lon_ny": ("lon", "lon+ny"),
    }
    return mapping.get(session_option, ("lon", None))


@router.get("/queue/status", response_model=QueueStatusResponse)
async def get_queue_status():
    """Get current queue status."""
    with _queue_lock:
        cards = []
        for card in _queue_state["cards"]:
            cards.append(TrainingCardStatus(**card["status"]))

        return QueueStatusResponse(
            queue_running=_queue_state["running"],
            current_card_id=_queue_state["current_card_id"],
            cards=cards,
        )


@router.post("/queue/add", response_model=AddToQueueResponse)
async def add_to_queue(request: AddToQueueRequest):
    """Add a training configuration card to the queue."""
    with _queue_lock:
        card_status = {
            "card_id": request.config.card_id,
            "status": "pending",
            "model_name": request.config.model_name,
            "current_epoch": 0,
            "total_epochs": request.config.num_epochs,
            "train_loss": None,
            "val_loss": None,
            "best_loss": None,
            "elapsed_seconds": 0.0,
            "directional_accuracy": None,
            "r_squared": None,
            "max_error": None,
            "grad_norm": None,
            "error_message": None,
            "final_directional_accuracy": None,
            "final_r_squared": None,
            "final_max_error": None,
        }
        _queue_state["cards"].append({
            "config": request.config.model_dump(),
            "status": card_status,
            "working_directory": request.working_directory,
        })

    return AddToQueueResponse(
        status="added",
        card_id=request.config.card_id,
        message=f"Added '{request.config.model_name}' to queue",
    )


@router.post("/queue/remove", response_model=RemoveFromQueueResponse)
async def remove_from_queue(request: RemoveFromQueueRequest):
    """Remove a card from the queue (only if pending)."""
    with _queue_lock:
        cards = _queue_state["cards"]
        for i, card in enumerate(cards):
            if card["status"]["card_id"] == request.card_id:
                if card["status"]["status"] == "pending":
                    cards.pop(i)
                    return RemoveFromQueueResponse(
                        status="removed",
                        card_id=request.card_id,
                        message="Card removed from queue",
                    )
                else:
                    return RemoveFromQueueResponse(
                        status="error",
                        card_id=request.card_id,
                        message="Cannot remove non-pending card",
                    )

    return RemoveFromQueueResponse(
        status="not_found",
        card_id=request.card_id,
        message="Card not found in queue",
    )


@router.post("/queue/start", response_model=StartQueueResponse)
async def start_queue(
    request: StartQueueRequest,
    background_tasks: BackgroundTasks,
):
    """Start processing the training queue sequentially."""
    with _queue_lock:
        if _queue_state["running"]:
            raise HTTPException(status_code=409, detail="Queue already running")

        # Check if there are pending cards
        pending_count = sum(1 for c in _queue_state["cards"] if c["status"]["status"] == "pending")
        if pending_count == 0:
            return StartQueueResponse(
                status="error",
                message="No pending cards in queue",
            )

        _queue_state["running"] = True

    background_tasks.add_task(_run_queue, request.working_directory)

    return StartQueueResponse(
        status="started",
        message=f"Queue started with {pending_count} pending card(s)",
    )


@router.post("/queue/stop", response_model=StopQueueResponse)
async def stop_queue():
    """Stop the queue after current training completes."""
    with _queue_lock:
        if not _queue_state["running"]:
            return StopQueueResponse(
                status="not_running",
                message="Queue is not running",
            )
        _queue_state["running"] = False

    # Also signal current training to stop
    with _training_lock:
        if _training_state["status"] == "training":
            _training_state["stop_requested"] = True

    return StopQueueResponse(
        status="stopping",
        message="Queue will stop after current training completes",
    )


@router.post("/queue/clear")
async def clear_queue():
    """Clear all completed/error cards from the queue."""
    with _queue_lock:
        _queue_state["cards"] = [
            c for c in _queue_state["cards"]
            if c["status"]["status"] in ("pending", "training")
        ]
    return {"status": "cleared", "message": "Cleared completed and error cards"}


def _run_queue(working_directory: Optional[str]):
    """Background task to process queue sequentially."""
    from app.transformer.trainer import TransformerTrainer
    from app.transformer.config import TransformerConfig

    while True:
        # Find next pending card
        with _queue_lock:
            if not _queue_state["running"]:
                break

            next_card = None
            for card in _queue_state["cards"]:
                if card["status"]["status"] == "pending":
                    next_card = card
                    break

            if not next_card:
                _queue_state["running"] = False
                break

            # Mark as training
            next_card["status"]["status"] = "training"
            _queue_state["current_card_id"] = next_card["status"]["card_id"]
            card_start_time = time.time()

        # Run training for this card
        try:
            config_data = next_card["config"]
            target_session, combine_sessions = _parse_session_option(config_data["session_option"])

            config = TransformerConfig(
                target_session=target_session,
                combine_sessions=combine_sessions,
                sequence_length=config_data["sequence_length"],
                batch_size=config_data["batch_size"],
                d_model=config_data["d_model"],
                n_layers=config_data["n_layers"],
                n_heads=config_data["n_heads"],
                dropout_rate=config_data["dropout_rate"],
                learning_rate=config_data["learning_rate"],
                num_epochs=config_data["num_epochs"],
                early_stopping_patience=config_data["early_stopping_patience"],
            )

            work_dir = Path(next_card.get("working_directory") or working_directory or ".")

            trainer = TransformerTrainer(
                config=config,
                working_directory=work_dir,
                model_name=config_data["model_name"],
            )

            # Use specific parquet file if provided
            parquet_filename = config_data.get("parquet_file")
            if parquet_filename:
                parquet_path = work_dir / "lstm" / "bridged" / parquet_filename
                if not parquet_path.exists():
                    raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
                trainer.setup(parquet_files=[parquet_path])

            # Callback to update card status
            def on_epoch_end(epoch, metrics):
                with _queue_lock:
                    if not _queue_state["running"]:
                        raise StopIteration("Queue stopped")

                    next_card["status"]["current_epoch"] = epoch + 1
                    next_card["status"]["train_loss"] = metrics.get("loss")
                    next_card["status"]["val_loss"] = metrics.get("val_loss")
                    next_card["status"]["grad_norm"] = metrics.get("grad_norm")
                    next_card["status"]["directional_accuracy"] = metrics.get("directional_accuracy")
                    next_card["status"]["r_squared"] = metrics.get("r_squared")
                    next_card["status"]["max_error"] = metrics.get("max_error")
                    next_card["status"]["elapsed_seconds"] = time.time() - card_start_time

                    if trainer.state:
                        next_card["status"]["best_loss"] = trainer.state.best_loss

            trainer.on_epoch_end = on_epoch_end

            # Set up immediate stop callback (checked after each batch)
            def check_should_stop():
                with _queue_lock:
                    return not _queue_state["running"]
            trainer.should_stop = check_should_stop

            state = trainer.train()

            # Export model if requested
            if config_data.get("save_to_models_folder", True):
                trainer.export_model(save_to_models_folder=True)

            # Mark completed with final metrics
            with _queue_lock:
                next_card["status"]["status"] = "completed"
                next_card["status"]["elapsed_seconds"] = time.time() - card_start_time
                next_card["status"]["final_directional_accuracy"] = state.final_directional_accuracy
                next_card["status"]["final_r_squared"] = state.final_r_squared
                next_card["status"]["final_max_error"] = state.final_max_error

        except StopIteration:
            with _queue_lock:
                next_card["status"]["status"] = "pending"  # Put back in queue
                next_card["status"]["error_message"] = "Stopped by user"
            break

        except Exception as e:
            with _queue_lock:
                next_card["status"]["status"] = "error"
                next_card["status"]["error_message"] = str(e)
                next_card["status"]["elapsed_seconds"] = time.time() - card_start_time

        finally:
            with _queue_lock:
                _queue_state["current_card_id"] = None

    with _queue_lock:
        _queue_state["running"] = False
        _queue_state["current_card_id"] = None
