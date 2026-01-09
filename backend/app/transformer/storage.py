"""
Storage utilities for Transformer models and checkpoints.

Manages paths for:
- Transformer model storage: {working_dir}/cache/transformer/models/
- Lists bridged parquet files from: {working_dir}/lstm/bridged/
"""

from pathlib import Path
from typing import Optional

from app.config import settings
from app.lstm.data_processor import get_lstm_path


def get_transformer_path(working_directory: Optional[Path] = None, create: bool = False) -> Path:
    """
    Get the transformer base directory path.

    Path: {working_dir}/cache/transformer/

    Args:
        working_directory: Base working directory
        create: If True, create the directory if it doesn't exist
    """
    base = working_directory or settings.default_working_directory
    transformer_path = base / "cache" / "transformer"
    if create:
        transformer_path.mkdir(parents=True, exist_ok=True)
    return transformer_path


def get_transformer_models_path(working_directory: Optional[Path] = None, create: bool = False) -> Path:
    """
    Get the transformer models directory path.

    Path: {working_dir}/cache/transformer/models/

    Args:
        working_directory: Base working directory
        create: If True, create the directory if it doesn't exist
    """
    models_path = get_transformer_path(working_directory, create=create) / "models"
    if create:
        models_path.mkdir(parents=True, exist_ok=True)
    return models_path


def list_bridged_parquet_files(working_directory: Optional[Path] = None) -> list[Path]:
    """
    List all bridged parquet files available for training.

    Searches: {working_dir}/lstm/bridged/
    """
    working_dir = str(working_directory) if working_directory else None
    bridged_path = get_lstm_path(working_dir, subfolder="bridged")
    return sorted(bridged_path.glob("*_bridged.parquet"))


def list_transformer_models(working_directory: Optional[Path] = None) -> list[str]:
    """
    List all available transformer model names.

    Scans for *_config.json files in the models directory.
    """
    models_path = get_transformer_models_path(working_directory, create=False)
    model_names = set()

    if not models_path.exists():
        return []

    for config_file in models_path.glob("*_config.json"):
        # Extract model name by removing _config suffix
        model_name = config_file.stem.replace("_config", "")
        model_names.add(model_name)

    return sorted(model_names)


def get_model_checkpoint_path(
    model_name: str,
    working_directory: Optional[Path] = None,
    best: bool = False,
) -> Path:
    """
    Get the path for a model checkpoint.

    Args:
        model_name: Name of the model
        working_directory: Base working directory
        best: If True, return path to best model, else latest checkpoint
    """
    models_path = get_transformer_models_path(working_directory)
    suffix = "_best.pt" if best else "_checkpoint.pt"
    return models_path / f"{model_name}{suffix}"


def get_model_config_path(
    model_name: str,
    working_directory: Optional[Path] = None,
) -> Path:
    """Get the path for a model's config JSON file."""
    models_path = get_transformer_models_path(working_directory)
    return models_path / f"{model_name}_config.json"


def get_model_state_path(
    model_name: str,
    working_directory: Optional[Path] = None,
) -> Path:
    """Get the path for a model's training state JSON file."""
    models_path = get_transformer_models_path(working_directory)
    return models_path / f"{model_name}_state.json"
