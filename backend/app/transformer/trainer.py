"""
Transformer Trainer for MMLC data.

Handles training loop, validation, checkpointing, and model saving.
"""

from pathlib import Path
from typing import Optional, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from app.transformer.config import TransformerConfig, TrainingState
from app.transformer.model import TimeStructureModel
from app.transformer.dataset import MMLCDataset, collate_mmlc, train_val_split_parquets
from app.transformer.storage import (
    get_transformer_models_path,
    list_bridged_parquet_files,
    get_model_checkpoint_path,
    get_model_config_path,
    get_model_state_path,
)


def generate_model_name(
    adr_period: int,
    bar_interval: str,
    session: str,
    custom_suffix: Optional[str] = None,
) -> str:
    """
    Generate default model name.

    Format: ADR{period}_{BarInterval}_{Session}[_{suffix}]
    Example: ADR20_M5_lon

    Args:
        adr_period: ADR calculation period (e.g., 20)
        bar_interval: Timeframe (e.g., "M5", "M15")
        session: Session prefix (e.g., "lon", "asia")
        custom_suffix: Optional custom suffix

    Returns:
        Model name string
    """
    name = f"ADR{adr_period}_{bar_interval}_{session}"
    if custom_suffix:
        name = f"{name}_{custom_suffix}"
    return name


class TransformerTrainer:
    """
    Trainer for the TimeStructureModel.

    Handles:
    - Data loading from enriched parquets
    - Training loop with MSE loss
    - Validation
    - Checkpointing
    - Early stopping
    - Model export
    """

    def __init__(
        self,
        config: TransformerConfig,
        working_directory: Optional[Path] = None,
        model_name: str = "transformer_v1",
    ):
        """
        Initialize trainer.

        Args:
            config: TransformerConfig instance
            working_directory: Base working directory
            model_name: Name for saving model files
        """
        self.config = config
        self.working_directory = working_directory
        self.model_name = model_name

        self.device = torch.device(config.get_device())
        print(f"Using device: {self.device}")

        # Paths - create directory since we'll be saving models
        self.models_path = get_transformer_models_path(working_directory, create=True)

        # Will be set in setup()
        self.model: Optional[TimeStructureModel] = None
        self.optimizer: Optional[AdamW] = None
        self.scheduler: Optional[ReduceLROnPlateau] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.state: Optional[TrainingState] = None

        # Callbacks
        self.on_epoch_end: Optional[Callable[[int, dict], None]] = None
        self.should_stop: Optional[Callable[[], bool]] = None  # Check after each batch for immediate abort

    def setup(
        self,
        parquet_files: Optional[list[Path]] = None,
        val_ratio: float = 0.2,
        seed: int = 42,
    ):
        """
        Set up model, data loaders, and optimizer.

        Args:
            parquet_files: List of enriched parquet file paths.
                           If None, auto-discovers from working directory.
            val_ratio: Fraction of files for validation
            seed: Random seed for split
        """
        # Auto-discover parquet files if not provided
        if parquet_files is None:
            parquet_files = list_bridged_parquet_files(self.working_directory)

        if not parquet_files:
            raise ValueError(
                "No parquet files provided and none found in lstm/bridged/. "
                "Run the LSTM bridge enrichment first."
            )

        print(f"Found {len(parquet_files)} parquet files")

        # Train/val split by file
        train_files, val_files = train_val_split_parquets(
            parquet_files, val_ratio, seed
        )
        print(f"Train: {len(train_files)} files, Val: {len(val_files)} files")

        # Create datasets
        train_datasets = []
        for fp in train_files:
            try:
                ds = MMLCDataset(
                    parquet_path=fp,
                    target_session=self.config.target_session,
                    sequence_length=self.config.sequence_length,
                    combine_sessions=self.config.combine_sessions,
                )
                if len(ds) > 0:
                    train_datasets.append(ds)
            except Exception as e:
                print(f"Warning: Failed to load {fp.name}: {e}")

        val_datasets = []
        for fp in val_files:
            try:
                ds = MMLCDataset(
                    parquet_path=fp,
                    target_session=self.config.target_session,
                    sequence_length=self.config.sequence_length,
                    combine_sessions=self.config.combine_sessions,
                )
                if len(ds) > 0:
                    val_datasets.append(ds)
            except Exception as e:
                print(f"Warning: Failed to load {fp.name}: {e}")

        if not train_datasets:
            raise ValueError("No valid training data found in parquet files.")
        if not val_datasets:
            raise ValueError("No valid validation data found in parquet files.")

        # Concatenate datasets
        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(val_datasets)

        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_mmlc,
            pin_memory=True,
            num_workers=0,  # Windows compatibility
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_mmlc,
            pin_memory=True,
            num_workers=0,
        )

        print(
            f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}"
        )

        # Create model
        self.model = TimeStructureModel.from_config(self.config).to(self.device)

        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {n_params:,}")

        # Optimizer
        self.optimizer = AdamW(
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

    def train_epoch(self) -> dict:
        """
        Train for one epoch.

        Returns:
            Dict with average loss and grad_norm for the epoch
        """
        self.model.train()
        total_loss = 0.0
        total_grad_norm = 0.0
        n_batches = 0

        criterion = nn.MSELoss()

        for batch in self.train_loader:
            x_cat = batch["x_cat"].to(self.device)
            x_cont = batch["x_cont"].to(self.device)
            y = batch["y"].to(self.device)
            mask = batch["mask"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            _, prediction = self.model(x_cat, x_cont, mask)

            # Compute loss
            loss = criterion(prediction.squeeze(-1), y)

            # Backward pass
            loss.backward()

            # Gradient clipping - capture the norm before clipping
            grad_norm = 0.0
            if self.config.grad_clip > 0:
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )
                if hasattr(grad_norm, 'item'):
                    grad_norm = grad_norm.item()
            total_grad_norm += grad_norm

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            # Check for immediate abort after each batch
            if self.should_stop and self.should_stop():
                raise StopIteration("Training stopped by user")

        return {
            "loss": total_loss / max(n_batches, 1),
            "grad_norm": total_grad_norm / max(n_batches, 1),
        }

    @torch.no_grad()
    def validate(self) -> dict:
        """
        Run validation.

        Returns:
            Dict with validation loss, directional_accuracy, r_squared, max_error
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        criterion = nn.MSELoss()

        # Collect predictions and targets for aggregate metrics
        all_predictions = []
        all_targets = []
        max_error = 0.0

        for batch in self.val_loader:
            x_cat = batch["x_cat"].to(self.device)
            x_cont = batch["x_cont"].to(self.device)
            y = batch["y"].to(self.device)
            mask = batch["mask"].to(self.device)

            _, prediction = self.model(x_cat, x_cont, mask)
            pred_squeezed = prediction.squeeze(-1)
            loss = criterion(pred_squeezed, y)

            total_loss += loss.item()
            n_batches += 1

            # Collect for aggregate metrics
            all_predictions.append(pred_squeezed.cpu())
            all_targets.append(y.cpu())

            # Track max error for this batch
            batch_max_error = (pred_squeezed - y).abs().max().item()
            max_error = max(max_error, batch_max_error)

        # Calculate aggregate metrics
        all_preds = torch.cat(all_predictions)
        all_targs = torch.cat(all_targets)

        # Directional accuracy: sign(pred) == sign(actual)
        pred_signs = torch.sign(all_preds)
        targ_signs = torch.sign(all_targs)
        directional_accuracy = (pred_signs == targ_signs).float().mean().item() * 100.0

        # R-squared: 1 - (SS_res / SS_tot)
        ss_res = ((all_targs - all_preds) ** 2).sum()
        ss_tot = ((all_targs - all_targs.mean()) ** 2).sum()
        r_squared = 1.0 - (ss_res / (ss_tot + 1e-8)).item()
        r_squared = max(0.0, min(1.0, r_squared))  # Clamp to valid range

        return {
            "loss": total_loss / max(n_batches, 1),
            "directional_accuracy": directional_accuracy,
            "r_squared": r_squared,
            "max_error": max_error,
        }

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_path = get_model_checkpoint_path(
            self.model_name, self.working_directory, best=False
        )
        best_path = get_model_checkpoint_path(
            self.model_name, self.working_directory, best=True
        )

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "state": self.state.to_dict(),
        }

        torch.save(checkpoint, checkpoint_path)

        if is_best:
            torch.save(checkpoint, best_path)

        # Save config/state as JSON for easy inspection
        config_path = get_model_config_path(self.model_name, self.working_directory)
        state_path = get_model_state_path(self.model_name, self.working_directory)

        self.config.save(config_path)
        self.state.save(state_path)

    def load_checkpoint(self, path: Optional[Path] = None) -> bool:
        """
        Load model checkpoint.

        Args:
            path: Checkpoint path. If None, loads latest checkpoint.

        Returns:
            True if checkpoint was loaded, False otherwise
        """
        if path is None:
            path = get_model_checkpoint_path(
                self.model_name, self.working_directory, best=False
            )

        if not path.exists():
            print(f"No checkpoint found at {path}")
            return False

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

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
            train_metrics = self.train_epoch()
            self.state.train_losses.append(train_metrics["loss"])
            self.state.grad_norms.append(train_metrics["grad_norm"])

            # Validate
            val_metrics = self.validate()
            self.state.val_losses.append(val_metrics["loss"])
            self.state.directional_accuracies.append(val_metrics["directional_accuracy"])
            self.state.r_squared_values.append(val_metrics["r_squared"])
            self.state.max_errors.append(val_metrics["max_error"])

            # Learning rate scheduling
            self.scheduler.step(val_metrics["loss"])
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Check for improvement
            is_best = val_metrics["loss"] < self.state.best_loss
            if is_best:
                self.state.best_loss = val_metrics["loss"]
                self.state.epochs_without_improvement = 0
            else:
                self.state.epochs_without_improvement += 1

            # Log progress with new metrics
            print(
                f"Epoch {epoch + 1:3d}/{self.config.num_epochs} | "
                f"Train: {train_metrics['loss']:.6f} | "
                f"Val: {val_metrics['loss']:.6f} | "
                f"DirAcc: {val_metrics['directional_accuracy']:.1f}% | "
                f"R²: {val_metrics['r_squared']:.3f} | "
                f"GradNorm: {train_metrics['grad_norm']:.2f} | "
                f"LR: {current_lr:.2e} | "
                f"{'*BEST*' if is_best else ''}"
            )

            # Callback - pass all metrics
            if self.on_epoch_end:
                self.on_epoch_end(epoch, {
                    **train_metrics,
                    "val_loss": val_metrics["loss"],
                    "directional_accuracy": val_metrics["directional_accuracy"],
                    "r_squared": val_metrics["r_squared"],
                    "max_error": val_metrics["max_error"],
                })

            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0 or is_best:
                self.save_checkpoint(is_best=is_best)

            # Early stopping
            if self.state.epochs_without_improvement >= self.config.early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        # Set final metrics from last validation
        if self.state.directional_accuracies:
            self.state.final_directional_accuracy = self.state.directional_accuracies[-1]
        if self.state.r_squared_values:
            self.state.final_r_squared = self.state.r_squared_values[-1]
        if self.state.max_errors:
            self.state.final_max_error = max(self.state.max_errors)

        print("=" * 60)
        print(f"Training complete! Best validation loss: {self.state.best_loss:.6f}")
        print(f"Final Dir Accuracy: {self.state.final_directional_accuracy:.1f}%")
        print(f"Final R²: {self.state.final_r_squared:.3f}")
        print(f"Max Error: {self.state.final_max_error:.4f}")

        # Save final checkpoint
        self.save_checkpoint()

        return self.state

    def export_model(
        self,
        path: Optional[Path] = None,
        save_to_models_folder: bool = False,
    ):
        """
        Export the best model for inference.

        Args:
            path: Output path (defaults to models/model_name.pt)
            save_to_models_folder: If True, also save to working_dir/models/
        """
        if path is None:
            path = self.models_path / f"{self.model_name}.pt"

        # Load best checkpoint
        best_path = get_model_checkpoint_path(
            self.model_name, self.working_directory, best=True
        )
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])

        # Save just the model for inference
        export_data = {
            "model_state_dict": self.model.state_dict(),
            "config": self.config.to_dict(),
            "latent_dim": self.config.latent_dim,
        }

        torch.save(export_data, path)
        print(f"Exported model to {path}")

        # Optionally save to user's models folder
        if save_to_models_folder and self.working_directory:
            user_models_path = self.working_directory / "models"
            user_models_path.mkdir(parents=True, exist_ok=True)
            user_model_path = user_models_path / f"{self.model_name}.pt"
            torch.save(export_data, user_model_path)
            print(f"Also saved to {user_model_path}")


def quick_train(
    parquet_files: Optional[list[Path]] = None,
    num_epochs: int = 10,
    working_directory: Optional[Path] = None,
    model_name: str = "transformer_test",
    target_session: str = "lon",
) -> TrainingState:
    """
    Quick training function for testing.

    Args:
        parquet_files: List of parquet files (or auto-discover)
        num_epochs: Number of epochs to train
        working_directory: Base directory
        model_name: Model name for saving
        target_session: Session to train on

    Returns:
        Training state
    """
    config = TransformerConfig(
        target_session=target_session,
        sequence_length=64,
        batch_size=32,
        d_model=128,
        n_layers=4,
        n_heads=4,
        num_epochs=num_epochs,
        learning_rate=1e-4,
        save_every=5,
        early_stopping_patience=10,
    )

    trainer = TransformerTrainer(
        config=config,
        working_directory=working_directory,
        model_name=model_name,
    )

    trainer.setup(parquet_files=parquet_files)
    return trainer.train()
