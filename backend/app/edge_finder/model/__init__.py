"""
VAE Model package for Edge Finder.

Contains the LSTM-based Variational Autoencoder for encoding
waveform matrices into latent vectors for pattern matching.
"""

from app.edge_finder.model.config import TrainingConfig
from app.edge_finder.model.dataset import WaveformDataset, create_dataloader
from app.edge_finder.model.vae import WaveformVAE
from app.edge_finder.model.trainer import VAETrainer

__all__ = [
    "TrainingConfig",
    "WaveformDataset",
    "create_dataloader",
    "WaveformVAE",
    "VAETrainer",
]
