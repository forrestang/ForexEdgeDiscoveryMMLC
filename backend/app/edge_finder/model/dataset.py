"""
PyTorch Dataset for waveform session data.

Handles loading session .npz files and creating batches
with proper padding for variable-length sequences.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from app.edge_finder.storage import list_session_files, get_sessions_path


class WaveformDataset(Dataset):
    """
    PyTorch Dataset for waveform matrices.

    Loads session .npz files and provides matrices with metadata.
    Handles variable-length sequences via collate function.
    """

    def __init__(
        self,
        session_ids: list[str],
        sessions_path: Path,
        min_bars: int = 20,
        max_bars: int = 300,
    ):
        """
        Args:
            session_ids: List of session IDs to include
            sessions_path: Path to sessions directory
            min_bars: Minimum sequence length (filter shorter)
            max_bars: Maximum sequence length (filter longer)
        """
        self.sessions_path = sessions_path
        self.min_bars = min_bars
        self.max_bars = max_bars

        # Filter sessions by length
        self.session_ids = []
        self.lengths = []

        for sid in session_ids:
            file_path = sessions_path / f"{sid}.npz"
            if file_path.exists():
                data = np.load(file_path)
                length = data["matrix"].shape[0]
                if min_bars <= length <= max_bars:
                    self.session_ids.append(sid)
                    self.lengths.append(length)

    def __len__(self) -> int:
        return len(self.session_ids)

    def __getitem__(self, idx: int) -> dict:
        """
        Load a single session.

        Returns:
            Dict with:
                - matrix: [seq_len, 20] tensor
                - length: int (original length)
                - session_id: str
                - metadata: dict with ground truth arrays
        """
        session_id = self.session_ids[idx]
        file_path = self.sessions_path / f"{session_id}.npz"

        data = np.load(file_path, allow_pickle=True)

        matrix = torch.tensor(data["matrix"], dtype=torch.float32)

        return {
            "matrix": matrix,
            "length": matrix.shape[0],
            "session_id": session_id,
            "next_bar_moves": torch.tensor(data["next_bar_moves"], dtype=torch.float32),
            "session_drifts": torch.tensor(data["session_drifts"], dtype=torch.float32),
            "maes": torch.tensor(data["maes"], dtype=torch.float32),
        }


def collate_waveforms(batch: list[dict]) -> dict:
    """
    Collate function for variable-length waveform sequences.

    Pads sequences to the maximum length in the batch and creates masks.

    Args:
        batch: List of dicts from WaveformDataset.__getitem__

    Returns:
        Dict with:
            - matrices: [batch_size, max_len, 20] padded tensor
            - lengths: [batch_size] tensor of original lengths
            - masks: [batch_size, max_len] boolean mask (True = valid)
            - session_ids: list of session IDs
    """
    # Sort by length (descending) for pack_padded_sequence efficiency
    batch = sorted(batch, key=lambda x: x["length"], reverse=True)

    matrices = [item["matrix"] for item in batch]
    lengths = torch.tensor([item["length"] for item in batch])
    session_ids = [item["session_id"] for item in batch]

    # Pad sequences
    padded = pad_sequence(matrices, batch_first=True, padding_value=0.0)

    # Create masks
    max_len = padded.shape[1]
    masks = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)

    return {
        "matrices": padded,
        "lengths": lengths,
        "masks": masks,
        "session_ids": session_ids,
    }


def create_dataloader(
    session_ids: list[str],
    sessions_path: Path,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    min_bars: int = 20,
    max_bars: int = 300,
) -> DataLoader:
    """
    Create a DataLoader for waveform data.

    Args:
        session_ids: List of session IDs to include
        sessions_path: Path to sessions directory
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers
        min_bars: Minimum sequence length
        max_bars: Maximum sequence length

    Returns:
        DataLoader with collate function for padding
    """
    dataset = WaveformDataset(
        session_ids=session_ids,
        sessions_path=sessions_path,
        min_bars=min_bars,
        max_bars=max_bars,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_waveforms,
        pin_memory=True,
    )


def train_val_split(
    session_ids: list[str],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """
    Split session IDs into train and validation sets.

    Args:
        session_ids: List of all session IDs
        val_ratio: Fraction for validation
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_ids, val_ids)
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(session_ids))

    n_val = int(len(session_ids) * val_ratio)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_ids = [session_ids[i] for i in train_indices]
    val_ids = [session_ids[i] for i in val_indices]

    return train_ids, val_ids
