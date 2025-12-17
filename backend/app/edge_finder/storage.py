"""
Storage module for Edge Finder session datasets.

Handles saving and loading session datasets in compressed .npz format.
Provides utilities for managing the edge_finder data directory.

Directory Structure:
    cache/edge_finder/
    ├── sessions/           # One .npz per session
    │   └── EURUSD_2024-01-15_london_M5.npz
    ├── models/             # Trained VAE weights (future)
    └── vectors/            # Latent vector index (future)
"""

from pathlib import Path
from datetime import date
from typing import Optional
import numpy as np

from app.edge_finder.future_truth import SessionDataset, BarMetadata
from app.config import settings


def get_edge_finder_path(working_directory: Optional[Path] = None) -> Path:
    """Get the edge_finder directory path, creating if needed."""
    base = working_directory or settings.default_working_directory
    edge_path = base / "cache" / "edge_finder"
    edge_path.mkdir(parents=True, exist_ok=True)
    return edge_path


def get_sessions_path(working_directory: Optional[Path] = None) -> Path:
    """Get the sessions directory path, creating if needed."""
    sessions_path = get_edge_finder_path(working_directory) / "sessions"
    sessions_path.mkdir(parents=True, exist_ok=True)
    return sessions_path


def get_models_path(working_directory: Optional[Path] = None) -> Path:
    """Get the models directory path, creating if needed."""
    models_path = get_edge_finder_path(working_directory) / "models"
    models_path.mkdir(parents=True, exist_ok=True)
    return models_path


def get_vectors_path(working_directory: Optional[Path] = None) -> Path:
    """Get the vectors directory path, creating if needed."""
    vectors_path = get_edge_finder_path(working_directory) / "vectors"
    vectors_path.mkdir(parents=True, exist_ok=True)
    return vectors_path


def save_session_dataset(
    dataset: SessionDataset,
    working_directory: Optional[Path] = None,
) -> Path:
    """
    Save a SessionDataset to compressed .npz format.

    File contents:
        - matrix: [num_bars, 20] float32
        - session_id: str
        - pair: str
        - session_date: str (ISO format)
        - session_type: str
        - timeframe: str
        - session_start_atr: float
        - total_bars: int
        - next_bar_moves: [num_bars] float32
        - session_drifts: [num_bars] float32
        - maes: [num_bars] float32
        - mfes: [num_bars] float32
        - bars_remaining: [num_bars] int32
        - session_progress: [num_bars] float32

    Args:
        dataset: SessionDataset to save
        working_directory: Optional override for base directory

    Returns:
        Path: Path to saved .npz file
    """
    sessions_path = get_sessions_path(working_directory)
    file_path = sessions_path / f"{dataset.session_id}.npz"

    # Extract metadata arrays
    n_bars = len(dataset.metadata)
    next_bar_moves = np.array([m.next_bar_move_atr for m in dataset.metadata], dtype=np.float32)
    session_drifts = np.array([m.session_drift_atr for m in dataset.metadata], dtype=np.float32)
    maes = np.array([m.mae_to_session_end for m in dataset.metadata], dtype=np.float32)
    mfes = np.array([m.mfe_to_session_end for m in dataset.metadata], dtype=np.float32)
    bars_remaining = np.array([m.bars_remaining for m in dataset.metadata], dtype=np.int32)
    session_progress = np.array([m.session_progress for m in dataset.metadata], dtype=np.float32)

    np.savez_compressed(
        file_path,
        matrix=dataset.matrix,
        session_id=dataset.session_id,
        pair=dataset.pair,
        session_date=dataset.session_date.isoformat(),
        session_type=dataset.session_type,
        timeframe=dataset.timeframe,
        session_start_atr=dataset.session_start_atr,
        total_bars=dataset.total_bars,
        next_bar_moves=next_bar_moves,
        session_drifts=session_drifts,
        maes=maes,
        mfes=mfes,
        bars_remaining=bars_remaining,
        session_progress=session_progress,
    )

    return file_path


def load_session_dataset(
    session_id: str,
    working_directory: Optional[Path] = None,
) -> Optional[SessionDataset]:
    """
    Load a SessionDataset from .npz file.

    Args:
        session_id: The session identifier (filename without extension)
        working_directory: Optional override for base directory

    Returns:
        SessionDataset or None if file doesn't exist
    """
    sessions_path = get_sessions_path(working_directory)
    file_path = sessions_path / f"{session_id}.npz"

    if not file_path.exists():
        return None

    data = np.load(file_path, allow_pickle=True)

    # Parse session date
    session_date_str = str(data["session_date"])
    session_date_parsed = date.fromisoformat(session_date_str)

    # Reconstruct metadata
    n_bars = int(data["total_bars"])
    metadata = []
    for i in range(n_bars):
        metadata.append(BarMetadata(
            session_id=str(data["session_id"]),
            bar_index=i,
            next_bar_move_atr=float(data["next_bar_moves"][i]),
            session_drift_atr=float(data["session_drifts"][i]),
            mae_to_session_end=float(data["maes"][i]),
            mfe_to_session_end=float(data["mfes"][i]),
            bars_remaining=int(data["bars_remaining"][i]),
            session_progress=float(data["session_progress"][i]),
        ))

    return SessionDataset(
        session_id=str(data["session_id"]),
        pair=str(data["pair"]),
        session_date=session_date_parsed,
        session_type=str(data["session_type"]),
        timeframe=str(data["timeframe"]),
        matrix=data["matrix"],
        metadata=metadata,
        session_start_atr=float(data["session_start_atr"]),
        total_bars=n_bars,
    )


def list_session_files(
    working_directory: Optional[Path] = None,
    pair: Optional[str] = None,
    session_type: Optional[str] = None,
    timeframe: Optional[str] = None,
) -> list[str]:
    """
    List all session dataset files, optionally filtered.

    Args:
        working_directory: Optional override for base directory
        pair: Filter by currency pair
        session_type: Filter by session type
        timeframe: Filter by timeframe

    Returns:
        List of session IDs (filenames without extension)
    """
    sessions_path = get_sessions_path(working_directory)

    session_ids = []
    for file_path in sessions_path.glob("*.npz"):
        session_id = file_path.stem

        # Apply filters
        if pair and not session_id.startswith(pair):
            continue
        if session_type and session_type not in session_id:
            continue
        if timeframe and not session_id.endswith(f"_{timeframe}"):
            continue

        session_ids.append(session_id)

    return sorted(session_ids)


def delete_session_file(
    session_id: str,
    working_directory: Optional[Path] = None,
) -> bool:
    """
    Delete a session dataset file.

    Args:
        session_id: The session identifier
        working_directory: Optional override for base directory

    Returns:
        True if file was deleted, False if it didn't exist
    """
    sessions_path = get_sessions_path(working_directory)
    file_path = sessions_path / f"{session_id}.npz"

    if file_path.exists():
        file_path.unlink()
        return True
    return False


def get_session_stats(
    working_directory: Optional[Path] = None,
) -> dict:
    """
    Get statistics about stored session datasets.

    Returns:
        Dict with counts by pair, session_type, timeframe
    """
    session_ids = list_session_files(working_directory)

    stats = {
        "total_sessions": len(session_ids),
        "by_pair": {},
        "by_session_type": {},
        "by_timeframe": {},
    }

    for sid in session_ids:
        parts = sid.split("_")
        if len(parts) >= 4:
            pair = parts[0]
            # Date is parts[1]
            session_type = parts[2]
            timeframe = parts[3]

            stats["by_pair"][pair] = stats["by_pair"].get(pair, 0) + 1
            stats["by_session_type"][session_type] = stats["by_session_type"].get(session_type, 0) + 1
            stats["by_timeframe"][timeframe] = stats["by_timeframe"].get(timeframe, 0) + 1

    return stats


# --- Model Management Functions ---

def list_models(working_directory: Optional[Path] = None) -> list[str]:
    """
    List all available model names.

    Returns:
        List of model names (without file extensions)
    """
    models_path = get_models_path(working_directory)
    model_names = set()

    for config_file in models_path.glob("*_config.json"):
        model_name = config_file.stem.replace("_config", "")
        model_names.add(model_name)

    return sorted(model_names)


def model_exists(model_name: str, working_directory: Optional[Path] = None) -> bool:
    """Check if a model with the given name exists."""
    models_path = get_models_path(working_directory)
    config_file = models_path / f"{model_name}_config.json"
    return config_file.exists()


def delete_model(model_name: str, working_directory: Optional[Path] = None) -> bool:
    """
    Delete a model and all its associated files.

    Deletes: {model_name}.pt, {model_name}_config.json, {model_name}_state.json,
             {model_name}_checkpoint.pt, {model_name}_best.pt

    Returns:
        True if any files were deleted, False if model didn't exist
    """
    models_path = get_models_path(working_directory)
    deleted = False

    suffixes = [".pt", "_config.json", "_state.json", "_checkpoint.pt", "_best.pt"]

    for suffix in suffixes:
        file_path = models_path / f"{model_name}{suffix}"
        if file_path.exists():
            file_path.unlink()
            deleted = True

    return deleted


def rename_model(
    old_name: str,
    new_name: str,
    working_directory: Optional[Path] = None,
) -> bool:
    """
    Rename a model and all its associated files.

    Returns:
        True if successful, False if old model doesn't exist or new name already exists
    """
    models_path = get_models_path(working_directory)

    # Check old exists
    if not model_exists(old_name, working_directory):
        return False

    # Check new doesn't exist
    if model_exists(new_name, working_directory):
        return False

    suffixes = [".pt", "_config.json", "_state.json", "_checkpoint.pt", "_best.pt"]

    for suffix in suffixes:
        old_path = models_path / f"{old_name}{suffix}"
        new_path = models_path / f"{new_name}{suffix}"
        if old_path.exists():
            old_path.rename(new_path)

    return True


def copy_model(
    source_name: str,
    dest_name: str,
    working_directory: Optional[Path] = None,
) -> bool:
    """
    Copy a model to a new name.

    Returns:
        True if successful, False if source doesn't exist or dest already exists
    """
    import shutil

    models_path = get_models_path(working_directory)

    # Check source exists
    if not model_exists(source_name, working_directory):
        return False

    # Check dest doesn't exist
    if model_exists(dest_name, working_directory):
        return False

    suffixes = [".pt", "_config.json", "_state.json", "_checkpoint.pt", "_best.pt"]

    for suffix in suffixes:
        source_path = models_path / f"{source_name}{suffix}"
        dest_path = models_path / f"{dest_name}{suffix}"
        if source_path.exists():
            shutil.copy2(source_path, dest_path)

    return True


# --- Index Management Functions ---

def list_indices(working_directory: Optional[Path] = None) -> list[str]:
    """
    List all available index names.

    Returns:
        List of index names (without file extensions)
    """
    vectors_path = get_vectors_path(working_directory)
    index_names = []

    for index_file in vectors_path.glob("*.npz"):
        index_names.append(index_file.stem)

    return sorted(index_names)


def index_exists(index_name: str, working_directory: Optional[Path] = None) -> bool:
    """Check if an index with the given name exists."""
    vectors_path = get_vectors_path(working_directory)
    index_file = vectors_path / f"{index_name}.npz"
    return index_file.exists()


def delete_index(index_name: str, working_directory: Optional[Path] = None) -> bool:
    """
    Delete an index file.

    Returns:
        True if deleted, False if didn't exist
    """
    vectors_path = get_vectors_path(working_directory)
    index_file = vectors_path / f"{index_name}.npz"

    if index_file.exists():
        index_file.unlink()
        return True
    return False
