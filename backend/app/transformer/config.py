"""
Configuration for Transformer training on MMLC data.

Supports loading from YAML config files for easy hyperparameter tuning.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json
import yaml


# Categorical ID mappings (0 reserved for padding)
LEVEL_TO_ID = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}  # 0 = padding
EVENT_TO_ID = {"SPAWN": 1, "EXTENSION": 2, "REVERSAL": 3}  # 0 = padding
DIRECTION_TO_ID = {"UP": 1, "DOWN": 2}  # 0 = padding

# Reverse mappings for inference
ID_TO_LEVEL = {v: k for k, v in LEVEL_TO_ID.items()}
ID_TO_EVENT = {v: k for k, v in EVENT_TO_ID.items()}
ID_TO_DIRECTION = {v: k for k, v in DIRECTION_TO_ID.items()}

# Session durations in hours
SESSION_HOURS = {
    "asia": 9,      # 00:00 - 09:00 UTC
    "lon": 9,       # 08:00 - 17:00 UTC
    "ny": 9,        # 13:00 - 22:00 UTC
    "day": 22,      # 00:00 - 22:00 UTC (full_day)
}

# Timeframe in minutes
TIMEFRAME_MINUTES = {
    "M1": 1,
    "M5": 5,
    "M10": 10,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
}


def get_default_sequence_length(
    session: str,
    timeframe: str,
    combine_sessions: Optional[str] = None,
) -> int:
    """
    Calculate the default sequence length (bar count) for a session/timeframe.

    Formula: bars = (session_hours × 60) ÷ timeframe_minutes

    Args:
        session: Target session (asia, lon, ny, day)
        timeframe: Bar timeframe (M1, M5, M10, M15, M30, H1, H4)
        combine_sessions: Optional combined sessions (e.g., "asia+lon")

    Returns:
        Number of bars in the session for the given timeframe

    Examples:
        >>> get_default_sequence_length("lon", "M5")
        108
        >>> get_default_sequence_length("lon", "M5", "asia+lon")
        216
        >>> get_default_sequence_length("day", "M15")
        88
    """
    minutes_per_bar = TIMEFRAME_MINUTES.get(timeframe, 5)

    if combine_sessions:
        # Sum hours for combined sessions (e.g., "asia+lon" = 9 + 9 = 18)
        prefixes = combine_sessions.split("+")
        total_hours = sum(SESSION_HOURS.get(p, 9) for p in prefixes)
    else:
        total_hours = SESSION_HOURS.get(session, 9)

    total_minutes = total_hours * 60
    return total_minutes // minutes_per_bar


def detect_sequence_length_from_parquet(
    parquet_path: Path,
    session: str,
) -> int:
    """
    Analyze parquet data to find optimal sequence length.

    Counts the actual number of bars per session in the data and returns
    the median, which is robust to outliers from incomplete sessions.

    Args:
        parquet_path: Path to the enriched parquet file
        session: Target session (asia, lon, ny, day)

    Returns:
        Median number of bars per session in the data
    """
    import polars as pl
    import numpy as np
    from app.transformer.validation import group_bars_into_sessions

    df = pl.read_parquet(parquet_path)

    # group_bars_into_sessions uses session time windows to group bars
    # The timeframe parameter is only used for logging, not filtering
    sessions = group_bars_into_sessions(df, session, "M5")

    if not sessions:
        print(f"  Warning: No sessions found in parquet, using default sequence_length=64")
        return 64  # Fallback default

    # Get bar counts per session
    bar_counts = [len(indices) for indices, _ in sessions]

    # Return median (robust to outliers from incomplete sessions)
    median_bars = int(np.median(bar_counts))
    print(f"  Session bar counts: min={min(bar_counts)}, median={median_bars}, max={max(bar_counts)}")

    return median_bars


@dataclass
class TransformerConfig:
    """
    Configuration for TimeStructureModel training.

    Supports loading from YAML config file for easy experimentation.
    """

    # === Data Settings ===
    target_session: str = "lon"  # Session prefix: asia, lon, ny, day
    combine_sessions: Optional[str] = None  # e.g., "asia+lon", "lon+ny"
    target_outcome: str = "max_up"  # Options: "max_up", "max_down", "next", "next5", "sess"
    sequence_length: int = 64  # Fixed window size
    batch_size: int = 32

    # === Model Dimensions ===
    d_model: int = 128  # Transformer hidden dimension
    n_layers: int = 4  # Number of encoder layers
    n_heads: int = 4  # Number of attention heads
    d_ff: int = 512  # Feed-forward dimension
    dropout_rate: float = 0.1

    # === Embedding Dimensions ===
    level_vocab_size: int = 6  # Levels 0-5 (0 for padding)
    event_vocab_size: int = 4  # 0=pad, 1=SPAWN, 2=EXT, 3=REV
    direction_vocab_size: int = 3  # 0=pad, 1=UP, 2=DOWN
    bar_position_vocab_size: int = 201  # 0-199 + padding token

    level_embed_dim: int = 16
    event_embed_dim: int = 8
    direction_embed_dim: int = 8
    bar_position_embed_dim: int = 32
    continuous_embed_dim: int = 8  # Projection dim for continuous input

    # === Latent and Output ===
    latent_dim: int = 128  # Latent vector dimension

    # === Training Controls ===
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    grad_clip: float = 1.0

    # === Checkpointing ===
    save_every: int = 10
    early_stopping_patience: int = 15

    # === Device ===
    device: str = "auto"  # "auto", "cuda", or "cpu"

    def get_device(self) -> str:
        """Get the actual device to use."""
        if self.device == "auto":
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    @property
    def total_embed_dim(self) -> int:
        """Total dimension of concatenated embeddings before projection."""
        return (
            self.level_embed_dim
            + self.event_embed_dim
            + self.direction_embed_dim
            + self.bar_position_embed_dim
            + self.continuous_embed_dim
        )

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            # Data
            "target_session": self.target_session,
            "combine_sessions": self.combine_sessions,
            "target_outcome": self.target_outcome,
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            # Model
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "dropout_rate": self.dropout_rate,
            # Embeddings
            "level_vocab_size": self.level_vocab_size,
            "event_vocab_size": self.event_vocab_size,
            "direction_vocab_size": self.direction_vocab_size,
            "bar_position_vocab_size": self.bar_position_vocab_size,
            "level_embed_dim": self.level_embed_dim,
            "event_embed_dim": self.event_embed_dim,
            "direction_embed_dim": self.direction_embed_dim,
            "bar_position_embed_dim": self.bar_position_embed_dim,
            "continuous_embed_dim": self.continuous_embed_dim,
            # Latent
            "latent_dim": self.latent_dim,
            # Training
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "num_epochs": self.num_epochs,
            "grad_clip": self.grad_clip,
            # Checkpointing
            "save_every": self.save_every,
            "early_stopping_patience": self.early_stopping_patience,
            # Device
            "device": self.device,
        }

    def save(self, path: Path):
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "TransformerConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: Path) -> "TransformerConfig":
        """
        Load configuration from YAML file.

        Supports nested structure:
        ```yaml
        data:
          target_session: "lon"
          sequence_length: 64
        model:
          d_model: 128
        training:
          learning_rate: 0.0001
        ```

        Or flat structure with direct keys.
        """
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Handle nested YAML structure
        flat_data = {}

        if "data" in data:
            flat_data.update(data["data"])
        if "model" in data:
            flat_data.update(data["model"])
        if "training" in data:
            flat_data.update(data["training"])
        if "embeddings" in data:
            flat_data.update(data["embeddings"])

        # Also include any top-level keys not in sections
        for key, value in data.items():
            if key not in ("data", "model", "training", "embeddings"):
                flat_data[key] = value

        # Filter to only valid config keys
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered_data = {k: v for k, v in flat_data.items() if k in valid_keys}

        return cls(**filtered_data)

    def save_yaml(self, path: Path):
        """Save config to YAML file with nested structure."""
        data = {
            "data": {
                "target_session": self.target_session,
                "combine_sessions": self.combine_sessions,
                "target_outcome": self.target_outcome,
                "sequence_length": self.sequence_length,
                "batch_size": self.batch_size,
            },
            "model": {
                "d_model": self.d_model,
                "n_layers": self.n_layers,
                "n_heads": self.n_heads,
                "d_ff": self.d_ff,
                "dropout_rate": self.dropout_rate,
                "latent_dim": self.latent_dim,
            },
            "training": {
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "num_epochs": self.num_epochs,
                "grad_clip": self.grad_clip,
                "early_stopping_patience": self.early_stopping_patience,
                "save_every": self.save_every,
            },
            "embeddings": {
                "level_vocab_size": self.level_vocab_size,
                "event_vocab_size": self.event_vocab_size,
                "direction_vocab_size": self.direction_vocab_size,
                "bar_position_vocab_size": self.bar_position_vocab_size,
                "level_embed_dim": self.level_embed_dim,
                "event_embed_dim": self.event_embed_dim,
                "direction_embed_dim": self.direction_embed_dim,
                "bar_position_embed_dim": self.bar_position_embed_dim,
                "continuous_embed_dim": self.continuous_embed_dim,
            },
            "device": self.device,
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


@dataclass
class TrainingState:
    """
    State of a training run (for resuming).
    """

    epoch: int = 0
    best_loss: float = float("inf")
    epochs_without_improvement: int = 0
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)

    # New metric histories
    directional_accuracies: list[float] = field(default_factory=list)
    r_squared_values: list[float] = field(default_factory=list)
    max_errors: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)

    # Final report metrics (populated when training completes)
    final_directional_accuracy: Optional[float] = None
    final_r_squared: Optional[float] = None
    final_max_error: Optional[float] = None
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "epochs_without_improvement": self.epochs_without_improvement,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "directional_accuracies": self.directional_accuracies,
            "r_squared_values": self.r_squared_values,
            "max_errors": self.max_errors,
            "grad_norms": self.grad_norms,
            "final_directional_accuracy": self.final_directional_accuracy,
            "final_r_squared": self.final_r_squared,
            "final_max_error": self.final_max_error,
            "elapsed_seconds": self.elapsed_seconds,
        }

    def save(self, path: Path):
        """Save state to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "TrainingState":
        """Load state from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)
