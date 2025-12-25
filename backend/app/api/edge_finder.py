"""
Edge Finder API endpoints.

Provides REST API for session generation, VAE training,
index building, and edge probability inference.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pathlib import Path
from typing import Optional
import numpy as np
import threading

from app.config import settings
from app.schemas.edge_finder import (
    GenerateSessionsRequest,
    GenerateSessionsResponse,
    SessionStats,
    TrainingRequest,
    TrainingStatus,
    TrainingResponse,
    ModelInfo,
    ModelListResponse,
    BuildIndexRequest,
    BuildIndexResponse,
    IndexStatus,
    IndexBuildingStatus,
    GenerationStatus,
    InferenceRequest,
    EdgeProbabilitiesResponse,
    InferenceResponse,
    ChartInferenceRequest,
    AutoSetupRequest,
    AutoSetupStatus,
    ModelRenameRequest,
    ModelCopyRequest,
    ModelActionResponse,
    DeleteResponse,
    FileListResponse,
    ParquetFileInfo,
    ModelSummary,
    IndexSummary,
    MatchDetailResponse,
    MineSessionRequest,
    MineSessionResponse,
    BarEdgeDataResponse,
)

router = APIRouter()

# --- Global State ---

_training_state = {
    "is_training": False,
    "epoch": 0,
    "total_epochs": 0,
    "train_loss": 0.0,
    "val_loss": 0.0,
    "best_loss": float("inf"),
    "progress": 0.0,
    "message": "Idle",
    "model_name": None,
    "stop_requested": False,
}

_last_training_result = {
    "completed": False,
    "model_name": None,
    "epochs_trained": 0,
    "best_loss": 0.0,
    "final_loss": 0.0,
}

_generation_state = {
    "is_generating": False,
    "progress": 0.0,
    "current_session": None,
    "message": "Idle",
}

_index_state = {
    "is_building": False,
    "current_session": 0,
    "total_sessions": 0,
    "progress": 0.0,
    "message": "Idle",
}

_loading_state = {
    "is_loading": False,
    "progress": 0.0,
    "message": "Idle",
    "model_name": None,
}

# Inference engine (lazy loaded)
_inference_engine = None
_inference_lock = threading.Lock()


def _get_working_dir(working_directory: Optional[str]) -> Path:
    """Get working directory, defaulting to settings."""
    if working_directory:
        return Path(working_directory)
    return settings.default_working_directory


# --- Session Generation Endpoints ---

@router.post("/sessions/generate", response_model=GenerateSessionsResponse)
async def generate_sessions(request: GenerateSessionsRequest, background_tasks: BackgroundTasks):
    """
    Generate session datasets from OHLC data.

    This processes raw OHLC data into waveform matrices with ground truth labels.
    """
    global _generation_state

    if _generation_state["is_generating"]:
        raise HTTPException(status_code=409, detail="Session generation already in progress")

    working_dir = _get_working_dir(request.working_directory)

    try:
        from app.edge_finder.generator import generate_test_dataset
        from app.edge_finder.storage import get_session_stats

        _generation_state["is_generating"] = True
        _generation_state["message"] = "Generating sessions..."

        # Generate sessions
        generated = generate_test_dataset(
            working_directory=working_dir,
            pair=request.pair or "EURUSD",
            session_type=request.session_type,
            timeframe=request.timeframe,
            max_sessions=request.max_sessions or 100,
        )

        _generation_state["is_generating"] = False
        _generation_state["message"] = "Complete"

        return GenerateSessionsResponse(
            status="success",
            sessions_generated=generated,
            errors=[],
            message=f"Generated {generated} session datasets",
        )

    except Exception as e:
        _generation_state["is_generating"] = False
        _generation_state["message"] = f"Error: {str(e)}"
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/stats", response_model=SessionStats)
async def get_session_stats_endpoint(working_directory: Optional[str] = None):
    """Get statistics about stored session datasets."""
    working_dir = _get_working_dir(working_directory)

    try:
        from app.edge_finder.storage import get_session_stats

        stats = get_session_stats(working_directory=working_dir)

        return SessionStats(
            total_sessions=stats["total_sessions"],
            by_pair=stats["by_pair"],
            by_session_type=stats["by_session_type"],
            by_timeframe=stats["by_timeframe"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/list")
async def list_sessions(
    working_directory: Optional[str] = None,
    pair: Optional[str] = None,
    session_type: Optional[str] = None,
    timeframe: Optional[str] = None,
    limit: int = 100,
):
    """List available session datasets."""
    working_dir = _get_working_dir(working_directory)

    try:
        from app.edge_finder.storage import list_session_files

        session_ids = list_session_files(
            working_directory=working_dir,
            pair=pair,
            session_type=session_type,
            timeframe=timeframe,
        )

        return {
            "total": len(session_ids),
            "sessions": session_ids[:limit],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Training Endpoints ---

@router.post("/training/start", response_model=TrainingResponse)
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Start VAE model training.

    Training runs in the background. Use /training/status to monitor progress.
    """
    global _training_state

    if _training_state["is_training"]:
        raise HTTPException(status_code=409, detail="Training already in progress")

    working_dir = _get_working_dir(request.working_directory)

    # Start training in background
    background_tasks.add_task(
        _run_training,
        working_dir=working_dir,
        model_name=request.model_name,
        num_epochs=request.num_epochs,
        batch_size=request.batch_size,
        learning_rate=request.learning_rate,
        latent_dim=request.latent_dim,
        kl_weight=request.kl_weight,
        pair=request.pair,
        session_type=request.session_type,
        timeframe=request.timeframe,
    )

    return TrainingResponse(
        status="started",
        message=f"Training started for model '{request.model_name}'",
    )


def _run_training(
    working_dir: Path,
    model_name: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    latent_dim: int,
    kl_weight: float,
    pair: Optional[str],
    session_type: Optional[str],
    timeframe: Optional[str],
):
    """Background task for VAE training."""
    global _training_state

    try:
        from app.edge_finder.model.trainer import VAETrainer
        from app.edge_finder.model.config import TrainingConfig

        _training_state["is_training"] = True
        _training_state["model_name"] = model_name
        _training_state["total_epochs"] = num_epochs
        _training_state["epoch"] = 0
        _training_state["stop_requested"] = False
        _training_state["message"] = "Initializing trainer..."

        config = TrainingConfig(
            latent_dim=latent_dim,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            kl_weight=kl_weight,
        )

        trainer = VAETrainer(
            config=config,
            working_directory=working_dir,
            model_name=model_name,
        )

        # Setup data loaders and model
        _training_state["message"] = "Loading session data..."
        trainer.setup()

        # Set up epoch callback for progress updates
        def on_epoch_end(epoch, metrics):
            _training_state["epoch"] = epoch + 1
            _training_state["train_loss"] = metrics.get("loss", 0.0)
            _training_state["val_loss"] = metrics.get("loss", 0.0)
            _training_state["progress"] = (epoch + 1) / num_epochs
            _training_state["message"] = f"Epoch {epoch + 1}/{num_epochs}"

            val_loss = metrics.get("loss", float("inf"))
            if val_loss < _training_state["best_loss"]:
                _training_state["best_loss"] = val_loss

        trainer.on_epoch_end = on_epoch_end

        # Run training
        trainer.train()

        _training_state["message"] = "Training complete"
        _training_state["progress"] = 1.0

        # Store training results for display after completion
        _last_training_result["completed"] = True
        _last_training_result["model_name"] = model_name
        _last_training_result["epochs_trained"] = _training_state["epoch"]
        _last_training_result["best_loss"] = _training_state["best_loss"] if _training_state["best_loss"] != float("inf") else 0.0
        _last_training_result["final_loss"] = _training_state["val_loss"]

    except Exception as e:
        _training_state["message"] = f"Error: {str(e)}"
        _last_training_result["completed"] = False

    finally:
        _training_state["is_training"] = False


def _run_index_build(
    working_dir: Path,
    model_name: str,
    pair: Optional[str],
    session_type: Optional[str],
    timeframe: Optional[str],
    save_index: bool = True,
    index_name: str = "latent_index",
):
    """Background task for building vector index with progress tracking."""
    global _index_state, _inference_engine

    try:
        from app.edge_finder.inference import EdgeInferenceEngine
        from app.edge_finder.storage import list_session_files

        _index_state["is_building"] = True
        _index_state["message"] = "Initializing..."
        _index_state["progress"] = 0.0

        # Count total sessions first
        session_ids = list_session_files(
            working_directory=working_dir,
            pair=pair,
            session_type=session_type,
            timeframe=timeframe,
        )
        _index_state["total_sessions"] = len(session_ids)

        # Progress callback
        def on_progress(current: int, total: int):
            _index_state["current_session"] = current
            _index_state["total_sessions"] = total
            _index_state["progress"] = current / total if total > 0 else 0
            _index_state["message"] = f"Encoding session {current}/{total}"

        # Create engine and build index
        engine = EdgeInferenceEngine(
            model_name=model_name,
            working_directory=working_dir,
            device="auto",
        )

        _index_state["message"] = "Loading model..."

        num_vectors = engine.initialize(
            load_saved_index=False,
            pair=pair,
            session_type=session_type,
            timeframe=timeframe,
            progress_callback=on_progress,
        )

        if save_index and num_vectors > 0:
            _index_state["message"] = "Saving index..."
            engine.save_index(index_name)

        with _inference_lock:
            _inference_engine = engine

        _index_state["message"] = "Complete"
        _index_state["progress"] = 1.0

    except Exception as e:
        _index_state["message"] = f"Error: {str(e)}"

    finally:
        _index_state["is_building"] = False


def _run_loading(
    working_dir: Path,
    model_name: str,
):
    """Background task for loading saved model and index."""
    global _loading_state, _inference_engine, _index_state

    try:
        from app.edge_finder.inference import EdgeInferenceEngine

        _loading_state["is_loading"] = True
        _loading_state["model_name"] = model_name
        _loading_state["message"] = "Loading model..."
        _loading_state["progress"] = 0.2

        engine = EdgeInferenceEngine(
            model_name=model_name,
            working_directory=working_dir,
            device="auto",
        )

        _loading_state["message"] = "Loading index..."
        _loading_state["progress"] = 0.5

        num_vectors = engine.initialize(
            load_saved_index=True,
            index_name="latent_index",
        )

        _loading_state["progress"] = 0.9

        if num_vectors > 0:
            with _inference_lock:
                _inference_engine = engine
            _loading_state["message"] = "Ready"
            _loading_state["progress"] = 1.0
        else:
            # No saved index found - need to build it
            _loading_state["is_loading"] = False
            _loading_state["message"] = "No saved index, building..."

            # Trigger index building
            _run_index_build(
                working_dir=working_dir,
                model_name=model_name,
                pair=None,
                session_type=None,
                timeframe=None,
            )
            return

    except Exception as e:
        _loading_state["message"] = f"Error: {str(e)}"

    finally:
        _loading_state["is_loading"] = False


@router.post("/training/stop", response_model=TrainingResponse)
async def stop_training():
    """Stop ongoing training."""
    global _training_state

    if not _training_state["is_training"]:
        raise HTTPException(status_code=400, detail="No training in progress")

    _training_state["stop_requested"] = True
    _training_state["message"] = "Stop requested..."

    return TrainingResponse(
        status="stopping",
        message="Training stop requested",
    )


@router.get("/training/status", response_model=TrainingStatus)
async def get_training_status():
    """Get current training status."""
    return TrainingStatus(
        is_training=_training_state["is_training"],
        epoch=_training_state["epoch"],
        total_epochs=_training_state["total_epochs"],
        train_loss=_training_state["train_loss"],
        val_loss=_training_state["val_loss"],
        best_loss=_training_state["best_loss"] if _training_state["best_loss"] != float("inf") else 0.0,
        progress=_training_state["progress"],
        message=_training_state["message"],
        model_name=_training_state["model_name"],
    )


# --- Status Endpoints ---

@router.get("/index/build/status", response_model=IndexBuildingStatus)
async def get_index_building_status():
    """Get current index building status."""
    return IndexBuildingStatus(
        is_building=_index_state["is_building"],
        current_session=_index_state["current_session"],
        total_sessions=_index_state["total_sessions"],
        progress=_index_state["progress"],
        message=_index_state["message"],
    )


@router.get("/sessions/generate/status", response_model=GenerationStatus)
async def get_generation_status():
    """Get current session generation status."""
    return GenerationStatus(
        is_generating=_generation_state["is_generating"],
        current_session=_generation_state["current_session"],
        progress=_generation_state["progress"],
        message=_generation_state["message"],
    )


# --- Model Endpoints ---

@router.get("/models", response_model=ModelListResponse)
async def list_models(working_directory: Optional[str] = None):
    """List available trained models."""
    working_dir = _get_working_dir(working_directory)

    try:
        from app.edge_finder.storage import get_models_path
        from app.edge_finder.model.config import TrainingConfig
        import json

        models_path = get_models_path(working_dir)
        models = []

        # Find all config files
        for config_file in models_path.glob("*_config.json"):
            model_name = config_file.stem.replace("_config", "")

            try:
                config = TrainingConfig.load(config_file)

                # Try to load state for training info
                state_file = models_path / f"{model_name}_state.json"
                trained_epochs = 0
                best_loss = float("inf")

                if state_file.exists():
                    with open(state_file, "r") as f:
                        state = json.load(f)
                        trained_epochs = state.get("epoch", 0)
                        best_loss = state.get("best_loss", float("inf"))

                # Calculate parameters (approximate)
                hidden_dim = config.hidden_dim
                latent_dim = config.latent_dim
                num_layers = config.num_layers
                # Rough parameter count
                total_params = (
                    20 * hidden_dim +  # input proj
                    4 * hidden_dim * hidden_dim * num_layers * 2 +  # LSTM encoder (bidirectional)
                    hidden_dim * 2 * latent_dim * 2 +  # mu/logvar
                    latent_dim * hidden_dim * num_layers * 2 +  # decoder init
                    4 * hidden_dim * hidden_dim * num_layers +  # LSTM decoder
                    hidden_dim * 20  # output proj
                )

                models.append(ModelInfo(
                    model_name=model_name,
                    latent_dim=config.latent_dim,
                    hidden_dim=config.hidden_dim,
                    num_layers=config.num_layers,
                    bidirectional=config.bidirectional,
                    total_parameters=total_params,
                    trained_epochs=trained_epochs,
                    best_loss=best_loss if best_loss != float("inf") else 0.0,
                ))

            except Exception:
                continue

        return ModelListResponse(models=models)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_name}", response_model=ModelInfo)
async def get_model_info(model_name: str, working_directory: Optional[str] = None):
    """Get information about a specific model."""
    working_dir = _get_working_dir(working_directory)

    try:
        from app.edge_finder.storage import get_models_path
        from app.edge_finder.model.config import TrainingConfig
        import json

        models_path = get_models_path(working_dir)
        config_file = models_path / f"{model_name}_config.json"

        if not config_file.exists():
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

        config = TrainingConfig.load(config_file)

        state_file = models_path / f"{model_name}_state.json"
        trained_epochs = 0
        best_loss = 0.0

        if state_file.exists():
            with open(state_file, "r") as f:
                state = json.load(f)
                trained_epochs = state.get("epoch", 0)
                best_loss = state.get("best_loss", 0.0)

        return ModelInfo(
            model_name=model_name,
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            bidirectional=config.bidirectional,
            total_parameters=0,  # Would need to load model to count
            trained_epochs=trained_epochs,
            best_loss=best_loss,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Index Endpoints ---

@router.post("/index/build", response_model=BuildIndexResponse)
async def build_index(request: BuildIndexRequest):
    """
    Build or rebuild the vector index from session data.

    This encodes all session matrices using the VAE and stores them for KNN search.
    """
    global _inference_engine

    working_dir = _get_working_dir(request.working_directory)

    try:
        from app.edge_finder.inference import EdgeInferenceEngine

        with _inference_lock:
            engine = EdgeInferenceEngine(
                model_name=request.model_name,
                working_directory=working_dir,
                device="auto",
            )

            # Build index
            num_vectors = engine.initialize(
                load_saved_index=False,
                pair=request.pair,
                session_type=request.session_type,
                timeframe=request.timeframe,
            )

            if request.save_index and num_vectors > 0:
                engine.save_index(request.index_name)

            # Store for inference
            _inference_engine = engine

        return BuildIndexResponse(
            status="success",
            num_vectors=num_vectors,
            num_sessions=engine.num_indexed_sessions,
            message=f"Index built with {num_vectors} vectors from {engine.num_indexed_sessions} sessions",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/index/status", response_model=IndexStatus)
async def get_index_status(working_directory: Optional[str] = None):
    """Get current index status."""
    global _inference_engine

    with _inference_lock:
        if _inference_engine is None or not _inference_engine.is_ready:
            return IndexStatus(
                is_loaded=False,
                num_vectors=0,
                num_sessions=0,
                latent_dim=32,
                model_name=None,
            )

        return IndexStatus(
            is_loaded=True,
            num_vectors=_inference_engine.num_indexed_vectors,
            num_sessions=_inference_engine.num_indexed_sessions,
            latent_dim=_inference_engine.index.latent_dim,
            model_name=_inference_engine.model_name,
        )


@router.post("/index/load", response_model=BuildIndexResponse)
async def load_index(
    working_directory: Optional[str] = None,
    model_name: str = "vae_test",
    index_name: str = "latent_index",
):
    """Load a previously saved index."""
    global _inference_engine

    working_dir = _get_working_dir(working_directory)

    try:
        from app.edge_finder.inference import EdgeInferenceEngine

        with _inference_lock:
            engine = EdgeInferenceEngine(
                model_name=model_name,
                working_directory=working_dir,
                device="auto",
            )

            num_vectors = engine.initialize(
                load_saved_index=True,
                index_name=index_name,
            )

            if num_vectors == 0:
                raise HTTPException(
                    status_code=404,
                    detail=f"Index '{index_name}' not found or empty",
                )

            _inference_engine = engine

        return BuildIndexResponse(
            status="success",
            num_vectors=num_vectors,
            num_sessions=engine.num_indexed_sessions,
            message=f"Index loaded with {num_vectors} vectors",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Inference Endpoints ---

@router.post("/inference", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest):
    """
    Run edge probability inference on a waveform matrix.

    Requires index to be loaded first via /index/build or /index/load.
    """
    global _inference_engine

    with _inference_lock:
        if _inference_engine is None or not _inference_engine.is_ready:
            raise HTTPException(
                status_code=400,
                detail="Index not loaded. Call /index/build or /index/load first.",
            )

        try:
            # Convert matrix
            matrix = np.array(request.matrix, dtype=np.float32)

            if matrix.ndim != 2 or matrix.shape[1] != 20:
                raise HTTPException(
                    status_code=400,
                    detail=f"Matrix must be [seq_len, 20], got {matrix.shape}",
                )

            # Run inference
            edge = _inference_engine.compute_edge(
                matrix=matrix,
                k=request.k_neighbors,
                unique_sessions=request.unique_sessions,
            )

            return InferenceResponse(
                status="success",
                edge=EdgeProbabilitiesResponse(
                    num_matches=edge.num_matches,
                    avg_distance=edge.avg_distance,
                    next_bar_up_pct=edge.next_bar_up_pct,
                    next_bar_avg_move=edge.next_bar_avg_move,
                    next_bar_std_move=edge.next_bar_std_move,
                    session_up_pct=edge.session_up_pct,
                    session_avg_drift=edge.session_avg_drift,
                    session_std_drift=edge.session_std_drift,
                    avg_mae=edge.avg_mae,
                    mae_p25=edge.mae_p25,
                    mae_p50=edge.mae_p50,
                    mae_p75=edge.mae_p75,
                    mae_p95=edge.mae_p95,
                    avg_mfe=edge.avg_mfe,
                    mfe_p25=edge.mfe_p25,
                    mfe_p50=edge.mfe_p50,
                    mfe_p75=edge.mfe_p75,
                    mfe_p95=edge.mfe_p95,
                    risk_reward_ratio=edge.risk_reward_ratio,
                    avg_session_progress=edge.avg_session_progress,
                    top_10_avg_distance=edge.top_10_avg_distance,
                    top_matches=[
                        MatchDetailResponse(
                            session_id=m.session_id,
                            bar_index=m.bar_index,
                            distance=m.distance,
                            next_bar_move=m.next_bar_move,
                            session_drift=m.session_drift,
                            mae=m.mae,
                            mfe=m.mfe,
                            session_progress=m.session_progress,
                        )
                        for m in edge.top_matches
                    ],
                ),
                message=f"Found {edge.num_matches} similar patterns",
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/inference/chart", response_model=InferenceResponse)
async def run_chart_inference(request: ChartInferenceRequest):
    """
    Run edge probability inference on a stored session (by chart identifiers).

    Loads the session matrix from storage and runs inference.
    Useful for getting edge probabilities for a displayed chart.
    """
    global _inference_engine

    with _inference_lock:
        if _inference_engine is None or not _inference_engine.is_ready:
            raise HTTPException(
                status_code=400,
                detail="Index not loaded. Call /index/build or /index/load first.",
            )

        try:
            from app.edge_finder.storage import load_session_dataset

            # Construct session_id from chart identifiers
            session_id = f"{request.pair}_{request.date}_{request.session}_{request.timeframe}"

            # Get working directory
            working_dir = Path(request.working_directory) if request.working_directory else None

            # Load the stored session
            session = load_session_dataset(session_id, working_dir)

            if session is None:
                return InferenceResponse(
                    status="not_found",
                    edge=None,
                    message=f"Session not found: {session_id}. Generate sessions first.",
                )

            # Get the matrix - use specified bar index or last bar
            matrix = session.matrix
            if request.bar_index is not None:
                if request.bar_index < 0 or request.bar_index >= len(matrix):
                    raise HTTPException(
                        status_code=400,
                        detail=f"bar_index {request.bar_index} out of range (0-{len(matrix)-1})",
                    )
                # Use matrix up to and including the specified bar
                matrix = matrix[:request.bar_index + 1]

            # Run inference
            edge = _inference_engine.compute_edge(
                matrix=matrix,
                k=request.k_neighbors,
                unique_sessions=request.unique_sessions,
            )

            return InferenceResponse(
                status="success",
                edge=EdgeProbabilitiesResponse(
                    num_matches=edge.num_matches,
                    avg_distance=edge.avg_distance,
                    next_bar_up_pct=edge.next_bar_up_pct,
                    next_bar_avg_move=edge.next_bar_avg_move,
                    next_bar_std_move=edge.next_bar_std_move,
                    session_up_pct=edge.session_up_pct,
                    session_avg_drift=edge.session_avg_drift,
                    session_std_drift=edge.session_std_drift,
                    avg_mae=edge.avg_mae,
                    mae_p25=edge.mae_p25,
                    mae_p50=edge.mae_p50,
                    mae_p75=edge.mae_p75,
                    mae_p95=edge.mae_p95,
                    avg_mfe=edge.avg_mfe,
                    mfe_p25=edge.mfe_p25,
                    mfe_p50=edge.mfe_p50,
                    mfe_p75=edge.mfe_p75,
                    mfe_p95=edge.mfe_p95,
                    risk_reward_ratio=edge.risk_reward_ratio,
                    avg_session_progress=edge.avg_session_progress,
                    top_10_avg_distance=edge.top_10_avg_distance,
                    top_matches=[
                        MatchDetailResponse(
                            session_id=m.session_id,
                            bar_index=m.bar_index,
                            distance=m.distance,
                            next_bar_move=m.next_bar_move,
                            session_drift=m.session_drift,
                            mae=m.mae,
                            mfe=m.mfe,
                            session_progress=m.session_progress,
                        )
                        for m in edge.top_matches
                    ],
                ),
                message=f"Found {edge.num_matches} similar patterns for {session_id}",
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


# --- Edge Mining Endpoints ---

@router.post("/mining/session", response_model=MineSessionResponse)
async def mine_session(request: MineSessionRequest):
    """
    Mine edge scores for ALL bars in a session simultaneously.

    Uses FAISS-accelerated batch KNN search to compute dual-horizon
    edge metrics for every bar position in the session.
    """
    global _inference_engine

    with _inference_lock:
        if _inference_engine is None or not _inference_engine.is_ready:
            raise HTTPException(
                status_code=400,
                detail="Index not loaded. Call /index/build or /index/load first.",
            )

        try:
            from app.edge_finder.storage import load_session_dataset
            from app.edge_finder.mining import EdgeMiner

            working_dir = _get_working_dir(request.working_directory)

            # Construct session_id from chart identifiers
            session_id = f"{request.pair}_{request.date}_{request.session}_{request.timeframe}"

            # Load the stored session
            session = load_session_dataset(session_id, working_dir)

            if session is None:
                return MineSessionResponse(
                    status="not_found",
                    graph_data=[],
                    edge_table=[],
                    message=f"Session not found: {session_id}. Generate sessions first.",
                )

            # Create miner and run
            miner = EdgeMiner(
                index=_inference_engine.index,
                k_neighbors=request.k_neighbors,
                use_faiss=True,
            )

            result = miner.mine_from_full_matrix(session.matrix)

            # Convert edge_table to response format
            edge_table_response = [
                BarEdgeDataResponse(
                    bar_index=e.bar_index,
                    next_bar_win_rate=e.next_bar_win_rate,
                    next_bar_avg_move=e.next_bar_avg_move,
                    next_bar_edge_score=e.next_bar_edge_score,
                    session_bias=e.session_bias,
                    session_win_rate=e.session_win_rate,
                    session_avg_mfe=e.session_avg_mfe,
                    session_avg_mae=e.session_avg_mae,
                    session_risk_reward=e.session_risk_reward,
                    session_edge_score=e.session_edge_score,
                    num_matches=e.num_matches,
                    avg_distance=e.avg_distance,
                )
                for e in result.edge_table
            ]

            return MineSessionResponse(
                status="success",
                graph_data=result.graph_data,
                edge_table=edge_table_response,
                message=f"Mined edge scores for {len(result.edge_table)} bars",
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check for edge finder service."""
    global _inference_engine

    return {
        "status": "healthy",
        "training_active": _training_state["is_training"],
        "index_loaded": _inference_engine is not None and _inference_engine.is_ready,
        "index_vectors": _inference_engine.num_indexed_vectors if _inference_engine and _inference_engine.is_ready else 0,
    }


# --- Auto Setup Endpoints ---

@router.post("/auto-setup", response_model=AutoSetupStatus)
async def auto_setup(request: AutoSetupRequest, background_tasks: BackgroundTasks):
    """
    Automatically set up model and index for inference.

    This smart endpoint:
    1. Checks if model exists -> if yes, uses it
    2. If no model or force_retrain -> starts training in background
    3. After model ready, checks index -> builds if needed
    4. Returns ready state when complete

    Use /auto-setup/status to monitor progress.
    """
    global _inference_engine, _training_state, _index_state

    working_dir = _get_working_dir(request.working_directory)

    # Check if already training
    if _training_state["is_training"]:
        return AutoSetupStatus(
            status="training",
            model_exists=False,
            model_name=_training_state["model_name"],
            index_loaded=False,
            num_vectors=0,
            num_sessions=0,
            message=_training_state["message"],
            training_epoch=_training_state["epoch"],
            training_total_epochs=_training_state["total_epochs"],
            training_loss=_training_state["val_loss"],
        )

    # Check if already building index
    if _index_state["is_building"]:
        return AutoSetupStatus(
            status="building_index",
            model_exists=True,
            model_name=request.model_name,
            index_loaded=False,
            num_vectors=0,
            num_sessions=0,
            message=_index_state["message"],
            index_current_session=_index_state["current_session"],
            index_total_sessions=_index_state["total_sessions"],
            index_progress=_index_state["progress"],
        )

    # Check if already loading
    if _loading_state["is_loading"]:
        return AutoSetupStatus(
            status="loading",
            model_exists=True,
            model_name=_loading_state["model_name"],
            index_loaded=False,
            num_vectors=0,
            num_sessions=0,
            message=_loading_state["message"],
            index_progress=_loading_state["progress"],
        )

    from app.edge_finder.storage import model_exists, get_session_stats

    # Check if model exists
    model_found = model_exists(request.model_name, working_dir)

    if model_found and not request.force_retrain:
        # Model exists, check/load index
        try:
            with _inference_lock:
                if _inference_engine is not None and _inference_engine.is_ready:
                    # Already loaded
                    return AutoSetupStatus(
                        status="ready",
                        model_exists=True,
                        model_name=request.model_name,
                        index_loaded=True,
                        num_vectors=_inference_engine.num_indexed_vectors,
                        num_sessions=_inference_engine.num_indexed_sessions,
                        message="Ready for inference",
                    )

            # Start loading in background
            background_tasks.add_task(
                _run_loading,
                working_dir=working_dir,
                model_name=request.model_name,
            )

            return AutoSetupStatus(
                status="loading",
                model_exists=True,
                model_name=request.model_name,
                index_loaded=False,
                num_vectors=0,
                num_sessions=0,
                message="Loading model and index...",
                index_progress=0.0,
            )

        except Exception as e:
            return AutoSetupStatus(
                status="error",
                model_exists=True,
                model_name=request.model_name,
                index_loaded=False,
                num_vectors=0,
                num_sessions=0,
                message=f"Error loading model: {str(e)}",
            )

    # Need to train - check if sessions exist
    stats = get_session_stats(working_dir)
    if stats["total_sessions"] == 0:
        return AutoSetupStatus(
            status="error",
            model_exists=False,
            model_name=request.model_name,
            index_loaded=False,
            num_vectors=0,
            num_sessions=0,
            message="No sessions found. Run pipeline first to generate session data.",
        )

    # Start training in background
    background_tasks.add_task(
        _run_training,
        working_dir=working_dir,
        model_name=request.model_name,
        num_epochs=request.num_epochs,
        batch_size=32,
        learning_rate=0.001,
        latent_dim=request.latent_dim,
        kl_weight=0.1,
        pair=request.pair,
        session_type=request.session_type,
        timeframe=request.timeframe,
    )

    return AutoSetupStatus(
        status="training",
        model_exists=False,
        model_name=request.model_name,
        index_loaded=False,
        num_vectors=0,
        num_sessions=stats["total_sessions"],
        message=f"Training started with {stats['total_sessions']} sessions",
    )


@router.get("/auto-setup/status", response_model=AutoSetupStatus)
async def get_auto_setup_status(working_directory: Optional[str] = None):
    """Get current status of auto-setup / inference readiness."""
    global _inference_engine, _training_state, _last_training_result, _index_state

    working_dir = _get_working_dir(working_directory)

    # Check if training
    if _training_state["is_training"]:
        return AutoSetupStatus(
            status="training",
            model_exists=False,
            model_name=_training_state["model_name"],
            index_loaded=False,
            num_vectors=0,
            num_sessions=0,
            message=_training_state["message"],
            training_epoch=_training_state["epoch"],
            training_total_epochs=_training_state["total_epochs"],
            training_loss=_training_state["val_loss"],
        )

    # Check if building index
    if _index_state["is_building"]:
        return AutoSetupStatus(
            status="building_index",
            model_exists=True,
            model_name=None,
            index_loaded=False,
            num_vectors=0,
            num_sessions=0,
            message=_index_state["message"],
            index_current_session=_index_state["current_session"],
            index_total_sessions=_index_state["total_sessions"],
            index_progress=_index_state["progress"],
        )

    # Check if inference ready
    with _inference_lock:
        if _inference_engine is not None and _inference_engine.is_ready:
            return AutoSetupStatus(
                status="ready",
                model_exists=True,
                model_name=_inference_engine.model_name,
                index_loaded=True,
                num_vectors=_inference_engine.num_indexed_vectors,
                num_sessions=_inference_engine.num_indexed_sessions,
                message="Ready for inference",
                last_training_completed=_last_training_result["completed"],
                last_training_best_loss=_last_training_result["best_loss"],
                last_training_epochs=_last_training_result["epochs_trained"],
            )

    # Check what we have
    from app.edge_finder.storage import list_models, get_session_stats

    models = list_models(working_dir)
    stats = get_session_stats(working_dir)

    if models:
        return AutoSetupStatus(
            status="checking",
            model_exists=True,
            model_name=models[0],
            index_loaded=False,
            num_vectors=0,
            num_sessions=stats["total_sessions"],
            message=f"Model '{models[0]}' available. Call auto-setup to load.",
        )

    if stats["total_sessions"] > 0:
        return AutoSetupStatus(
            status="checking",
            model_exists=False,
            model_name=None,
            index_loaded=False,
            num_vectors=0,
            num_sessions=stats["total_sessions"],
            message=f"{stats['total_sessions']} sessions available. Call auto-setup to train model.",
        )

    return AutoSetupStatus(
        status="error",
        model_exists=False,
        model_name=None,
        index_loaded=False,
        num_vectors=0,
        num_sessions=0,
        message="No sessions found. Run pipeline first.",
    )


# --- Model Management Endpoints ---

@router.post("/models/{model_name}/rename", response_model=ModelActionResponse)
async def rename_model_endpoint(
    model_name: str,
    request: ModelRenameRequest,
):
    """Rename a model."""
    working_dir = _get_working_dir(request.working_directory)

    from app.edge_finder.storage import rename_model

    success = rename_model(model_name, request.new_name, working_dir)

    if success:
        return ModelActionResponse(
            success=True,
            message=f"Model renamed from '{model_name}' to '{request.new_name}'",
        )
    else:
        return ModelActionResponse(
            success=False,
            message=f"Failed to rename. Model '{model_name}' may not exist or '{request.new_name}' already exists.",
        )


@router.post("/models/{model_name}/copy", response_model=ModelActionResponse)
async def copy_model_endpoint(
    model_name: str,
    request: ModelCopyRequest,
):
    """Copy/save-as a model."""
    working_dir = _get_working_dir(request.working_directory)

    from app.edge_finder.storage import copy_model

    success = copy_model(model_name, request.new_name, working_dir)

    if success:
        return ModelActionResponse(
            success=True,
            message=f"Model copied from '{model_name}' to '{request.new_name}'",
        )
    else:
        return ModelActionResponse(
            success=False,
            message=f"Failed to copy. Model '{model_name}' may not exist or '{request.new_name}' already exists.",
        )


# --- Bulk Delete Endpoints ---
# NOTE: These static paths MUST be defined BEFORE dynamic path routes
# to prevent FastAPI from matching "all" as a path parameter.

@router.delete("/parquets/all", response_model=DeleteResponse)
async def delete_all_parquets(working_directory: Optional[str] = None):
    """Delete all parquet cache files."""
    working_dir = _get_working_dir(working_directory)
    cache_path = working_dir / "cache"

    deleted = 0
    if cache_path.exists():
        for file_path in cache_path.glob("*.parquet"):
            file_path.unlink()
            deleted += 1

    return DeleteResponse(
        success=True,
        deleted_count=deleted,
        message=f"Deleted {deleted} parquet files",
    )


@router.delete("/sessions/all", response_model=DeleteResponse)
async def delete_all_sessions(working_directory: Optional[str] = None):
    """Delete all session dataset files."""
    working_dir = _get_working_dir(working_directory)

    from app.edge_finder.storage import get_sessions_path

    sessions_path = get_sessions_path(working_dir)
    deleted = 0
    for file_path in sessions_path.glob("*.npz"):
        file_path.unlink()
        deleted += 1

    return DeleteResponse(
        success=True,
        deleted_count=deleted,
        message=f"Deleted {deleted} session files",
    )


@router.delete("/models/all", response_model=DeleteResponse)
async def delete_all_models(working_directory: Optional[str] = None):
    """Delete all model files."""
    working_dir = _get_working_dir(working_directory)

    from app.edge_finder.storage import list_models, delete_model

    model_names = list_models(working_dir)
    deleted = 0
    for model_name in model_names:
        if delete_model(model_name, working_dir):
            deleted += 1

    return DeleteResponse(
        success=True,
        deleted_count=deleted,
        message=f"Deleted {deleted} models",
    )


@router.delete("/indices/all", response_model=DeleteResponse)
async def delete_all_indices(working_directory: Optional[str] = None):
    """Delete all vector index files."""
    working_dir = _get_working_dir(working_directory)

    from app.edge_finder.storage import list_indices, delete_index

    index_names = list_indices(working_dir)
    deleted = 0
    for index_name in index_names:
        if delete_index(index_name, working_dir):
            deleted += 1

    return DeleteResponse(
        success=True,
        deleted_count=deleted,
        message=f"Deleted {deleted} indices",
    )


@router.delete("/all", response_model=DeleteResponse)
async def delete_all_files(working_directory: Optional[str] = None):
    """Delete all generated files (parquets, sessions, models, indices). Does NOT delete CSV source files."""
    working_dir = _get_working_dir(working_directory)

    from app.edge_finder.storage import (
        get_sessions_path,
        list_models,
        delete_model,
        list_indices,
        delete_index,
    )

    total_deleted = 0

    # Delete parquets
    cache_path = working_dir / "cache"
    if cache_path.exists():
        for file_path in cache_path.glob("*.parquet"):
            file_path.unlink()
            total_deleted += 1

    # Delete sessions
    sessions_path = get_sessions_path(working_dir)
    for file_path in sessions_path.glob("*.npz"):
        file_path.unlink()
        total_deleted += 1

    # Delete models
    for model_name in list_models(working_dir):
        if delete_model(model_name, working_dir):
            total_deleted += 1

    # Delete indices
    for index_name in list_indices(working_dir):
        if delete_index(index_name, working_dir):
            total_deleted += 1

    return DeleteResponse(
        success=True,
        deleted_count=total_deleted,
        message=f"Deleted {total_deleted} files total (parquets, sessions, models, indices)",
    )


# --- Individual Delete Endpoints ---
# NOTE: These dynamic path routes MUST be defined AFTER static paths like /parquets/all
# to prevent FastAPI from matching "all" as a path parameter.

@router.delete("/parquets/{pair}/{timeframe}", response_model=ModelActionResponse)
async def delete_parquet_endpoint(
    pair: str,
    timeframe: str,
    working_directory: Optional[str] = None,
):
    """Delete a parquet cache file."""
    working_dir = _get_working_dir(working_directory)

    cache_path = working_dir / "cache"
    file_path = cache_path / f"{pair}_{timeframe}.parquet"

    if file_path.exists():
        file_path.unlink()
        return ModelActionResponse(
            success=True,
            message=f"Deleted parquet file: {pair}_{timeframe}.parquet",
        )
    else:
        return ModelActionResponse(
            success=False,
            message=f"Parquet file not found: {pair}_{timeframe}.parquet",
        )


@router.delete("/sessions", response_model=DeleteResponse)
async def delete_sessions_endpoint(
    pair: str,
    session_type: str,
    timeframe: str,
    working_directory: Optional[str] = None,
):
    """
    Bulk delete sessions matching filter criteria.

    All three filter parameters are REQUIRED to prevent accidental mass deletion.
    """
    working_dir = _get_working_dir(working_directory)

    from app.edge_finder.storage import list_session_files, delete_session_file

    # Get matching sessions
    session_ids = list_session_files(
        working_directory=working_dir,
        pair=pair,
        session_type=session_type,
        timeframe=timeframe,
    )

    if not session_ids:
        return DeleteResponse(
            success=True,
            deleted_count=0,
            message="No sessions matched the filter criteria",
        )

    # Delete each session
    deleted = 0
    for session_id in session_ids:
        if delete_session_file(session_id, working_dir):
            deleted += 1

    return DeleteResponse(
        success=True,
        deleted_count=deleted,
        message=f"Deleted {deleted} session files matching {pair}/{session_type}/{timeframe}",
    )


@router.delete("/models/{model_name}", response_model=ModelActionResponse)
async def delete_model_endpoint(
    model_name: str,
    working_directory: Optional[str] = None,
):
    """Delete a model."""
    working_dir = _get_working_dir(working_directory)

    from app.edge_finder.storage import delete_model

    success = delete_model(model_name, working_dir)

    if success:
        return ModelActionResponse(
            success=True,
            message=f"Model '{model_name}' deleted",
        )
    else:
        return ModelActionResponse(
            success=False,
            message=f"Model '{model_name}' not found",
        )


@router.delete("/indices/{index_name}", response_model=ModelActionResponse)
async def delete_index_endpoint(
    index_name: str,
    working_directory: Optional[str] = None,
):
    """Delete a vector index file."""
    working_dir = _get_working_dir(working_directory)

    from app.edge_finder.storage import delete_index

    success = delete_index(index_name, working_dir)

    if success:
        return ModelActionResponse(
            success=True,
            message=f"Index '{index_name}' deleted",
        )
    else:
        return ModelActionResponse(
            success=False,
            message=f"Index '{index_name}' not found",
        )


# --- File Listing Endpoint ---

@router.get("/files", response_model=FileListResponse)
async def list_all_files(working_directory: Optional[str] = None):
    """
    Get comprehensive listing of all generated files.

    Returns parquets, sessions, models, and indices.
    """
    global _inference_engine

    working_dir = _get_working_dir(working_directory)

    from app.edge_finder.storage import (
        get_session_stats,
        list_models,
        list_indices,
        get_models_path,
        get_vectors_path,
    )
    from app.core.cache_manager import get_cached_instruments

    cache_path = working_dir / "cache"

    # Get parquets
    parquets = []
    try:
        instruments = get_cached_instruments(cache_path)
        for inst in instruments:
            for tf in inst.get("timeframes", []):
                file_path = cache_path / f"{inst['pair']}_{tf}.parquet"
                if file_path.exists():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    parquets.append(ParquetFileInfo(
                        pair=inst["pair"],
                        timeframe=tf,
                        file_name=file_path.name,
                        size_mb=round(size_mb, 2),
                    ))
    except Exception:
        pass

    # Get session stats
    stats = get_session_stats(working_dir)

    # Get models
    models = []
    model_names = list_models(working_dir)
    models_path = get_models_path(working_dir)

    active_model = None
    with _inference_lock:
        if _inference_engine is not None and _inference_engine.is_ready:
            active_model = _inference_engine.model_name

    for name in model_names:
        try:
            import json
            config_file = models_path / f"{name}_config.json"
            state_file = models_path / f"{name}_state.json"

            latent_dim = 32
            trained_epochs = 0
            best_loss = 0.0

            if config_file.exists():
                with open(config_file, "r") as f:
                    config = json.load(f)
                    latent_dim = config.get("latent_dim", 32)

            if state_file.exists():
                with open(state_file, "r") as f:
                    state = json.load(f)
                    trained_epochs = state.get("epoch", 0)
                    best_loss = state.get("best_loss", 0.0)

            models.append(ModelSummary(
                model_name=name,
                latent_dim=latent_dim,
                trained_epochs=trained_epochs,
                best_loss=best_loss,
                is_active=(name == active_model),
            ))
        except Exception:
            models.append(ModelSummary(
                model_name=name,
                latent_dim=0,
                trained_epochs=0,
                best_loss=0.0,
                is_active=(name == active_model),
            ))

    # Get indices
    indices = []
    index_names = list_indices(working_dir)
    vectors_path = get_vectors_path(working_dir)

    for name in index_names:
        try:
            index_file = vectors_path / f"{name}.npz"
            data = np.load(index_file, allow_pickle=True)
            num_vectors = len(data.get("vectors", []))
            model_name = str(data.get("model_name", "unknown"))
            indices.append(IndexSummary(
                index_name=name,
                num_vectors=num_vectors,
                model_name=model_name,
            ))
        except Exception:
            indices.append(IndexSummary(
                index_name=name,
                num_vectors=0,
                model_name=None,
            ))

    return FileListResponse(
        parquets=parquets,
        sessions_total=stats["total_sessions"],
        sessions_by_pair=stats["by_pair"],
        sessions_by_session_type=stats["by_session_type"],
        sessions_by_timeframe=stats["by_timeframe"],
        models=models,
        indices=indices,
    )
