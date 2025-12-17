"""
Test script for VAE training (Phase 5 validation).

Run from backend directory:
    python test_vae_training.py
"""

import sys
sys.path.insert(0, '.')

import torch
from pathlib import Path

from app.edge_finder.storage import list_session_files, get_session_stats
from app.edge_finder.model.config import TrainingConfig
from app.edge_finder.model.trainer import VAETrainer


def main():
    print("=" * 60)
    print("VAE Training Test - Phase 5 Validation")
    print("=" * 60)

    # Check available sessions
    session_stats = get_session_stats()
    print(f"\nAvailable sessions: {session_stats['total_sessions']}")
    print(f"By pair: {session_stats['by_pair']}")

    if session_stats['total_sessions'] == 0:
        print("ERROR: No sessions found. Run test_edge_finder.py first.")
        return

    # Quick training config for validation
    config = TrainingConfig(
        # Model
        input_channels=20,
        latent_dim=32,
        hidden_dim=128,  # Smaller for faster testing
        num_layers=2,
        bidirectional=True,
        dropout=0.1,

        # Training - small for testing
        batch_size=16,
        learning_rate=1e-3,
        num_epochs=10,  # Just 10 epochs for validation
        kl_weight=0.1,
        kl_annealing=True,
        kl_annealing_epochs=3,

        # Data
        min_bars=20,
        max_bars=300,

        # Checkpointing
        save_every=5,
        early_stopping_patience=10,

        # Device
        device="auto",
    )

    print(f"\nDevice: {config.get_device()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Create trainer
    print("\n" + "-" * 40)
    print("Setting up trainer...")
    print("-" * 40)

    trainer = VAETrainer(
        config=config,
        model_name="vae_test",
    )

    trainer.setup(val_ratio=0.2, seed=42)

    # Show model summary
    print(f"\nModel architecture:")
    print(f"  Encoder: Bidirectional LSTM")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Latent dim: {config.latent_dim}")
    print(f"  Num layers: {config.num_layers}")

    # Train
    print("\n" + "-" * 40)
    print("Starting training...")
    print("-" * 40)

    state = trainer.train()

    # Results
    print("\n" + "-" * 40)
    print("Training Results:")
    print("-" * 40)
    print(f"  Final epoch: {state.epoch + 1}")
    print(f"  Best val loss: {state.best_loss:.4f}")
    print(f"  Train losses: {[f'{l:.4f}' for l in state.train_losses[-5:]]}")
    print(f"  Val losses: {[f'{l:.4f}' for l in state.val_losses[-5:]]}")

    # Test encoding
    print("\n" + "-" * 40)
    print("Testing encoding...")
    print("-" * 40)

    trainer.model.eval()

    # Get a sample batch
    for batch in trainer.val_loader:
        matrices = batch["matrices"].to(trainer.device)
        lengths = batch["lengths"]

        with torch.no_grad():
            # Encode to latent space
            z = trainer.model.encode(matrices, lengths)

        print(f"  Input shape: {matrices.shape}")
        print(f"  Latent shape: {z.shape}")
        print(f"  Latent mean: {z.mean().item():.4f}")
        print(f"  Latent std: {z.std().item():.4f}")
        print(f"  Latent min: {z.min().item():.4f}")
        print(f"  Latent max: {z.max().item():.4f}")
        break

    # Export model
    print("\n" + "-" * 40)
    print("Exporting model...")
    print("-" * 40)

    trainer.export_model()

    # List saved files
    models_path = trainer.models_path
    print(f"\nSaved files in {models_path}:")
    for f in models_path.glob("vae_test*"):
        print(f"  {f.name}")

    print("\n" + "=" * 60)
    print("Phase 5 Validation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
