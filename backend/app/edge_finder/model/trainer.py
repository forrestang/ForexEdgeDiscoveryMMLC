"""
VAE Trainer for waveform data.

Handles training loop, validation, checkpointing, and early stopping.
"""

from pathlib import Path
from typing import Optional, Callable
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from app.edge_finder.model.config import TrainingConfig, TrainingState
from app.edge_finder.model.vae import WaveformVAE, vae_loss
from app.edge_finder.model.dataset import (
    create_dataloader,
    train_val_split,
)
from app.edge_finder.storage import get_sessions_path, get_models_path, list_session_files


class VAETrainer:
    """
    Trainer for the WaveformVAE model.

    Handles the complete training pipeline including:
    - Data loading
    - Training loop
    - Validation
    - Checkpointing
    - Early stopping
    - Metrics logging
    """

    def __init__(
        self,
        config: TrainingConfig,
        working_directory: Optional[Path] = None,
        model_name: str = "vae_v1",
    ):
        """
        Initialize trainer.

        Args:
            config: Training configuration
            working_directory: Base directory for data and models
            model_name: Name for saving model files
        """
        self.config = config
        self.working_directory = working_directory
        self.model_name = model_name

        self.device = torch.device(config.get_device())
        print(f"Using device: {self.device}")

        # Paths
        self.sessions_path = get_sessions_path(working_directory)
        self.models_path = get_models_path(working_directory)

        # Will be set in setup()
        self.model: Optional[WaveformVAE] = None
        self.optimizer: Optional[Adam] = None
        self.scheduler: Optional[ReduceLROnPlateau] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.state: Optional[TrainingState] = None

        # Callbacks
        self.on_epoch_end: Optional[Callable[[int, dict], None]] = None

    def setup(self, val_ratio: float = 0.2, seed: int = 42):
        """
        Set up model, data loaders, and optimizer.

        Args:
            val_ratio: Fraction of data for validation
            seed: Random seed for train/val split
        """
        # Get all session files
        session_ids = list_session_files(self.working_directory)
        if not session_ids:
            raise ValueError("No session files found. Run data generation first.")

        print(f"Found {len(session_ids)} sessions")

        # Train/val split
        train_ids, val_ids = train_val_split(session_ids, val_ratio, seed)
        print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")

        # Create data loaders
        self.train_loader = create_dataloader(
            session_ids=train_ids,
            sessions_path=self.sessions_path,
            batch_size=self.config.batch_size,
            shuffle=True,
            min_bars=self.config.min_bars,
            max_bars=self.config.max_bars,
        )

        self.val_loader = create_dataloader(
            session_ids=val_ids,
            sessions_path=self.sessions_path,
            batch_size=self.config.batch_size,
            shuffle=False,
            min_bars=self.config.min_bars,
            max_bars=self.config.max_bars,
        )

        print(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")

        # Create model
        self.model = WaveformVAE(
            input_dim=self.config.input_channels,
            hidden_dim=self.config.hidden_dim,
            latent_dim=self.config.latent_dim,
            num_layers=self.config.num_layers,
            bidirectional=self.config.bidirectional,
            dropout=self.config.dropout,
        ).to(self.device)

        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {n_params:,}")

        # Optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        # Training state
        self.state = TrainingState()

    def get_kl_weight(self, epoch: int) -> float:
        """Get KL weight with optional annealing."""
        if not self.config.kl_annealing:
            return self.config.kl_weight

        if epoch < self.config.kl_annealing_epochs:
            # Linear annealing from 0 to kl_weight
            return self.config.kl_weight * (epoch / self.config.kl_annealing_epochs)
        return self.config.kl_weight

    def train_epoch(self, epoch: int) -> dict:
        """
        Train for one epoch.

        Returns:
            Dict with average metrics for the epoch
        """
        self.model.train()
        total_metrics = {"loss": 0, "recon_loss": 0, "kl_loss": 0}
        n_batches = 0

        kl_weight = self.get_kl_weight(epoch)

        for batch in self.train_loader:
            matrices = batch["matrices"].to(self.device)
            lengths = batch["lengths"]
            masks = batch["masks"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            recon, mu, logvar = self.model(matrices, lengths)

            # Compute loss
            loss, metrics = vae_loss(
                recon=recon,
                target=matrices,
                mu=mu,
                logvar=logvar,
                mask=masks,
                kl_weight=kl_weight,
            )

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip,
                )

            self.optimizer.step()

            # Accumulate metrics
            for key in total_metrics:
                total_metrics[key] += metrics[key]
            n_batches += 1

        # Average metrics
        for key in total_metrics:
            total_metrics[key] /= n_batches

        total_metrics["kl_weight"] = kl_weight

        return total_metrics

    @torch.no_grad()
    def validate(self) -> dict:
        """
        Run validation.

        Returns:
            Dict with average validation metrics
        """
        self.model.eval()
        total_metrics = {"loss": 0, "recon_loss": 0, "kl_loss": 0}
        n_batches = 0

        for batch in self.val_loader:
            matrices = batch["matrices"].to(self.device)
            lengths = batch["lengths"]
            masks = batch["masks"].to(self.device)

            # Forward pass
            recon, mu, logvar = self.model(matrices, lengths)

            # Compute loss (full KL weight for validation)
            _, metrics = vae_loss(
                recon=recon,
                target=matrices,
                mu=mu,
                logvar=logvar,
                mask=masks,
                kl_weight=self.config.kl_weight,
            )

            # Accumulate metrics
            for key in total_metrics:
                total_metrics[key] += metrics[key]
            n_batches += 1

        # Average metrics
        for key in total_metrics:
            total_metrics[key] /= n_batches

        return total_metrics

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_path = self.models_path / f"{self.model_name}_checkpoint.pt"
        best_path = self.models_path / f"{self.model_name}_best.pt"

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "state": self.state.to_dict(),
        }

        torch.save(checkpoint, checkpoint_path)

        if is_best:
            torch.save(checkpoint, best_path)

        # Also save config as JSON for easy inspection
        self.config.save(self.models_path / f"{self.model_name}_config.json")
        self.state.save(self.models_path / f"{self.model_name}_state.json")

    def load_checkpoint(self, path: Optional[Path] = None):
        """Load model checkpoint."""
        if path is None:
            path = self.models_path / f"{self.model_name}_checkpoint.pt"

        if not path.exists():
            print(f"No checkpoint found at {path}")
            return False

        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "state" in checkpoint:
            state_dict = checkpoint["state"]
            self.state = TrainingState(**state_dict)

        print(f"Loaded checkpoint from epoch {self.state.epoch}")
        return True

    def train(self, resume: bool = False) -> TrainingState:
        """
        Run full training loop.

        Args:
            resume: If True, try to resume from checkpoint

        Returns:
            Final training state
        """
        if self.model is None:
            self.setup()

        if resume:
            self.load_checkpoint()

        print(f"\nStarting training for {self.config.num_epochs} epochs")
        print("=" * 60)

        start_epoch = self.state.epoch

        for epoch in range(start_epoch, self.config.num_epochs):
            self.state.epoch = epoch

            # Train
            train_metrics = self.train_epoch(epoch)
            self.state.train_losses.append(train_metrics["loss"])

            # Validate
            val_metrics = self.validate()
            self.state.val_losses.append(val_metrics["loss"])

            # Learning rate scheduling
            self.scheduler.step(val_metrics["loss"])

            # Check for improvement
            is_best = val_metrics["loss"] < self.state.best_loss
            if is_best:
                self.state.best_loss = val_metrics["loss"]
                self.state.epochs_without_improvement = 0
            else:
                self.state.epochs_without_improvement += 1

            # Log progress
            print(
                f"Epoch {epoch + 1:3d}/{self.config.num_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Recon: {val_metrics['recon_loss']:.4f} | "
                f"KL: {val_metrics['kl_loss']:.4f} | "
                f"{'*BEST*' if is_best else ''}"
            )

            # Callback
            if self.on_epoch_end:
                self.on_epoch_end(epoch, {**train_metrics, **val_metrics})

            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0 or is_best:
                self.save_checkpoint(is_best=is_best)

            # Early stopping
            if self.state.epochs_without_improvement >= self.config.early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        print("=" * 60)
        print(f"Training complete! Best validation loss: {self.state.best_loss:.4f}")

        # Save final checkpoint
        self.save_checkpoint()

        return self.state

    def export_model(self, path: Optional[Path] = None):
        """
        Export the best model for inference.

        Args:
            path: Output path (defaults to models/model_name.pt)
        """
        if path is None:
            path = self.models_path / f"{self.model_name}.pt"

        # Load best checkpoint
        best_path = self.models_path / f"{self.model_name}_best.pt"
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])

        # Save just the model for inference
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config.to_dict(),
            "latent_dim": self.config.latent_dim,
        }, path)

        print(f"Exported model to {path}")


def quick_train(
    num_epochs: int = 10,
    working_directory: Optional[Path] = None,
    model_name: str = "vae_test",
) -> TrainingState:
    """
    Quick training function for testing.

    Args:
        num_epochs: Number of epochs to train
        working_directory: Base directory
        model_name: Model name for saving

    Returns:
        Training state
    """
    config = TrainingConfig(
        num_epochs=num_epochs,
        batch_size=16,
        learning_rate=1e-3,
        kl_weight=0.1,
        kl_annealing=True,
        kl_annealing_epochs=5,
        save_every=5,
        early_stopping_patience=10,
    )

    trainer = VAETrainer(
        config=config,
        working_directory=working_directory,
        model_name=model_name,
    )

    trainer.setup()
    return trainer.train()
