"""
Transformer package for MMLC time-series prediction.

Contains the encoder-only Transformer model for predicting
forex outcomes from MMLC state sequences.
"""

from app.transformer.config import (
    TransformerConfig,
    TrainingState,
    get_default_sequence_length,
    SESSION_HOURS,
    TIMEFRAME_MINUTES,
)
from app.transformer.dataset import MMLCDataset, create_dataloader, collate_mmlc
from app.transformer.model import TimeStructureModel
from app.transformer.trainer import TransformerTrainer, generate_model_name
from app.transformer.storage import (
    get_transformer_path,
    get_transformer_models_path,
    list_bridged_parquet_files,
    list_transformer_models,
)
from app.transformer.validation import (
    generate_sanity_check,
    generate_memory_test,
    run_validation,
    run_all_validations,
)

__all__ = [
    "TransformerConfig",
    "TrainingState",
    "get_default_sequence_length",
    "SESSION_HOURS",
    "TIMEFRAME_MINUTES",
    "MMLCDataset",
    "create_dataloader",
    "collate_mmlc",
    "TimeStructureModel",
    "TransformerTrainer",
    "generate_model_name",
    "get_transformer_path",
    "get_transformer_models_path",
    "list_bridged_parquet_files",
    "list_transformer_models",
    "generate_sanity_check",
    "generate_memory_test",
    "run_validation",
    "run_all_validations",
]
