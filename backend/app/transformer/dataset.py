"""
PyTorch Dataset for MMLC enriched parquet data.

Handles:
- Loading enriched parquet files with Polars
- Session-based filtering (asia, lon, ny, day)
- Combined session support (asia+lon, lon+ny)
- Fixed-length windowing with left-padding
- Normalization of price deltas by session ADR
"""

import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional
import numpy as np

from app.transformer.config import LEVEL_TO_ID, EVENT_TO_ID, DIRECTION_TO_ID


# Session prefix to ADR column mapping
SESSION_TO_ADR = {
    "asia": "adr_asia",
    "lon": "adr_london",
    "ny": "adr_ny",
    "day": "adr_full_day",
    "asialon": "adr_asia_london",
    "lonny": "adr_london_ny",
}

# Valid session prefixes
VALID_SESSIONS = {"asia", "lon", "ny", "day", "asialon", "lonny"}


class MMLCDataset(Dataset):
    """
    PyTorch Dataset for MMLC-enriched parquet files.

    Features:
    - Session-based indexing with prefix filtering
    - Combined sessions support (consecutive sessions like "asia+lon")
    - Left-padding with zeros for sequences shorter than window
    - Price delta normalization by session ADR

    Input tensors:
        x_cat: (seq_len, 4) - Level, Event, Direction, BarPosition IDs
        x_cont: (seq_len, 1) - Normalized price delta

    Target:
        y: scalar - {prefix}_out_max_up value
    """

    def __init__(
        self,
        parquet_path: Path,
        target_session: str = "lon",
        sequence_length: int = 64,
        combine_sessions: Optional[str] = None,
    ):
        """
        Args:
            parquet_path: Path to enriched parquet file
            target_session: Session prefix (asia, lon, ny, day)
            sequence_length: Fixed window size (left-padded if shorter)
            combine_sessions: Optional combined sessions (e.g., "asia+lon")
        """
        self.parquet_path = Path(parquet_path)
        self.target_session = target_session
        self.sequence_length = sequence_length
        self.combine_sessions = combine_sessions

        # Validate target session
        if target_session not in VALID_SESSIONS:
            raise ValueError(
                f"Invalid target_session '{target_session}'. "
                f"Must be one of: {VALID_SESSIONS}"
            )

        # Load parquet with Polars
        self.df = pl.read_parquet(self.parquet_path)

        # Determine session prefixes to include
        self.session_prefixes = self._parse_session_prefixes()

        # Get ADR column name
        self.adr_column = SESSION_TO_ADR[self.target_session]

        # Build index of valid sample positions
        self._build_sample_index()

    def _parse_session_prefixes(self) -> list[str]:
        """Parse session prefixes from target_session or combine_sessions."""
        if self.combine_sessions:
            # e.g., "asia+lon" -> ["asia", "lon"]
            prefixes = self.combine_sessions.split("+")
            for p in prefixes:
                if p not in VALID_SESSIONS:
                    raise ValueError(
                        f"Invalid session '{p}' in combine_sessions. "
                        f"Must be one of: {VALID_SESSIONS}"
                    )
            return prefixes
        return [self.target_session]

    def _build_sample_index(self):
        """
        Build index of valid row indices for sampling.

        A row is valid if:
        - Target session has non-null state_level
        - For combined sessions: ALL sessions have non-null state_level
        - Has non-null outcome value
        """
        prefix = self.target_session
        state_col = f"{prefix}_state_level"
        outcome_col = f"{prefix}_out_max_up"

        # Start with rows where target session has valid state and outcome
        valid_mask = self.df[state_col].is_not_null() & self.df[outcome_col].is_not_null()

        # For combined sessions, require all sessions to be valid
        if self.combine_sessions:
            for p in self.session_prefixes:
                valid_mask = valid_mask & self.df[f"{p}_state_level"].is_not_null()

        # Get row indices where valid
        valid_indices = np.where(valid_mask.to_numpy())[0]
        self.sample_indices = valid_indices.tolist()

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample.

        Returns:
            dict with:
                x_cat: (seq_len, 4) tensor - categorical features
                x_cont: (seq_len, 1) tensor - continuous features
                y: scalar tensor - target outcome
                mask: (seq_len,) boolean mask (True = valid, False = padding)
        """
        end_idx = self.sample_indices[idx]
        start_idx = max(0, end_idx - self.sequence_length + 1)
        actual_len = end_idx - start_idx + 1

        # Extract window from DataFrame
        window_df = self.df.slice(start_idx, actual_len)

        prefix = self.target_session

        # Extract categorical features
        levels = window_df[f"{prefix}_state_level"].fill_null(0).to_numpy()
        events = window_df[f"{prefix}_state_event"].to_numpy()
        directions = window_df[f"{prefix}_state_dir"].to_numpy()

        # Map to IDs (0 for null/unknown values)
        level_ids = np.array([LEVEL_TO_ID.get(int(l) if l else 0, 0) for l in levels], dtype=np.int64)
        event_ids = np.array([EVENT_TO_ID.get(e, 0) if e else 0 for e in events], dtype=np.int64)
        direction_ids = np.array([DIRECTION_TO_ID.get(d, 0) if d else 0 for d in directions], dtype=np.int64)

        # Bar position: index within window (0 to max 199)
        bar_positions = np.clip(np.arange(actual_len), 0, 199).astype(np.int64)

        # Extract continuous features
        opens = window_df["open"].to_numpy()
        closes = window_df["close"].to_numpy()
        adrs = window_df[self.adr_column].fill_null(1.0).to_numpy()

        # Normalize price delta: (Close - Open) / session_adr
        # Prevent division by zero
        adrs = np.where(adrs == 0, 1.0, adrs)
        price_deltas = (closes - opens) / adrs

        # Left-pad to sequence_length
        pad_len = self.sequence_length - actual_len

        if pad_len > 0:
            level_ids = np.pad(level_ids, (pad_len, 0), constant_values=0)
            event_ids = np.pad(event_ids, (pad_len, 0), constant_values=0)
            direction_ids = np.pad(direction_ids, (pad_len, 0), constant_values=0)
            bar_positions = np.pad(bar_positions, (pad_len, 0), constant_values=0)
            price_deltas = np.pad(price_deltas, (pad_len, 0), constant_values=0.0)

        # Stack categorical features: (seq_len, 4)
        x_cat = np.stack([level_ids, event_ids, direction_ids, bar_positions], axis=1)
        x_cont = price_deltas.reshape(-1, 1).astype(np.float32)

        # Get target: {prefix}_out_max_up at the sample position
        target_col = f"{prefix}_out_max_up"
        y = self.df[end_idx, target_col]
        if y is None:
            y = 0.0

        # Create mask (True = valid, False = padding)
        mask = np.zeros(self.sequence_length, dtype=bool)
        mask[pad_len:] = True

        return {
            "x_cat": torch.tensor(x_cat, dtype=torch.long),
            "x_cont": torch.tensor(x_cont, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
        }


def collate_mmlc(batch: list[dict]) -> dict:
    """
    Collate function for MMLCDataset.

    Stacks batch items into tensors.
    """
    x_cat = torch.stack([item["x_cat"] for item in batch])
    x_cont = torch.stack([item["x_cont"] for item in batch])
    y = torch.stack([item["y"] for item in batch])
    mask = torch.stack([item["mask"] for item in batch])

    return {
        "x_cat": x_cat,  # (batch, seq_len, 4)
        "x_cont": x_cont,  # (batch, seq_len, 1)
        "y": y,  # (batch,)
        "mask": mask,  # (batch, seq_len)
    }


def create_dataloader(
    parquet_path: Path,
    target_session: str = "lon",
    sequence_length: int = 64,
    batch_size: int = 32,
    shuffle: bool = True,
    combine_sessions: Optional[str] = None,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader for MMLC data.

    Args:
        parquet_path: Path to enriched parquet file
        target_session: Session prefix (asia, lon, ny, day)
        sequence_length: Fixed window size
        batch_size: Batch size
        shuffle: Whether to shuffle data
        combine_sessions: Optional combined sessions (e.g., "asia+lon")
        num_workers: Number of data loading workers

    Returns:
        PyTorch DataLoader
    """
    dataset = MMLCDataset(
        parquet_path=parquet_path,
        target_session=target_session,
        sequence_length=sequence_length,
        combine_sessions=combine_sessions,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_mmlc,
        num_workers=num_workers,
        pin_memory=True,
    )


def train_val_split_parquets(
    parquet_files: list[Path],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[Path], list[Path]]:
    """
    Split parquet files into train and validation sets.

    Splits by file (not by sample) to avoid data leakage between
    temporally adjacent samples.

    Args:
        parquet_files: List of parquet file paths
        val_ratio: Fraction of files for validation
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_files, val_files)
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(parquet_files))

    n_val = max(1, int(len(parquet_files) * val_ratio))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_files = [parquet_files[i] for i in train_indices]
    val_files = [parquet_files[i] for i in val_indices]

    return train_files, val_files
