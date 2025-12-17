"""
Training configuration for the VAE model.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json


@dataclass
class TrainingConfig:
    """
    Configuration for VAE training.

    All hyperparameters and paths for training the waveform VAE.
    """
    # Model architecture
    input_channels: int = 20          # Number of features per timestep
    latent_dim: int = 32              # Latent space dimensionality
    hidden_dim: int = 256             # LSTM hidden dimension
    num_layers: int = 2               # Number of LSTM layers
    bidirectional: bool = True        # Use bidirectional LSTM
    dropout: float = 0.1              # Dropout rate

    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 100
    kl_weight: float = 0.1            # Beta for beta-VAE (KL term weight)
    kl_annealing: bool = True         # Gradually increase KL weight
    kl_annealing_epochs: int = 20     # Epochs to anneal KL weight

    # Data filtering
    min_bars: int = 20                # Minimum session length to include
    max_bars: int = 300               # Maximum session length

    # Regularization
    weight_decay: float = 1e-5
    grad_clip: float = 1.0            # Gradient clipping

    # Checkpointing
    save_every: int = 10              # Save checkpoint every N epochs
    early_stopping_patience: int = 15 # Stop if no improvement for N epochs

    # Device
    device: str = "auto"              # "auto", "cuda", or "cpu"

    def get_device(self) -> str:
        """Get the actual device to use."""
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "input_channels": self.input_channels,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "bidirectional": self.bidirectional,
            "dropout": self.dropout,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "kl_weight": self.kl_weight,
            "kl_annealing": self.kl_annealing,
            "kl_annealing_epochs": self.kl_annealing_epochs,
            "min_bars": self.min_bars,
            "max_bars": self.max_bars,
            "weight_decay": self.weight_decay,
            "grad_clip": self.grad_clip,
            "save_every": self.save_every,
            "early_stopping_patience": self.early_stopping_patience,
            "device": self.device,
        }

    def save(self, path: Path):
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "TrainingConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


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

    def to_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "epochs_without_improvement": self.epochs_without_improvement,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
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
