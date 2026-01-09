"""
TimeStructureModel - Encoder-only Transformer for MMLC prediction.

Architecture:
- Separate nn.Embedding for each categorical: Level, Event, Direction, BarPosition
- Linear projection for continuous input
- Concatenate embeddings + projected continuous, project to d_model
- nn.TransformerEncoder backbone
- Last-token pooling
- MLP projection to latent vector (dim=128)
- Linear head -> 1 for outcome prediction
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TimeStructureModel(nn.Module):
    """
    Encoder-only Transformer for MMLC time-series prediction.

    Input:
        x_cat: (batch, seq_len, 4) - Level, Event, Direction, BarPosition IDs
        x_cont: (batch, seq_len, 1) - Normalized price delta
        mask: (batch, seq_len) - Attention mask (True = valid)

    Output:
        latent: (batch, latent_dim) - Latent representation
        prediction: (batch, 1) - Outcome prediction
    """

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        # Embedding vocab sizes
        level_vocab_size: int = 6,
        event_vocab_size: int = 4,
        direction_vocab_size: int = 3,
        bar_position_vocab_size: int = 201,
        # Embedding dimensions
        level_embed_dim: int = 16,
        event_embed_dim: int = 8,
        direction_embed_dim: int = 8,
        bar_position_embed_dim: int = 32,
        continuous_embed_dim: int = 8,
        # Output
        latent_dim: int = 128,
        max_seq_len: int = 500,
    ):
        super().__init__()

        self.d_model = d_model
        self.latent_dim = latent_dim

        # === Embedding Layers ===
        self.level_embed = nn.Embedding(
            level_vocab_size, level_embed_dim, padding_idx=0
        )
        self.event_embed = nn.Embedding(
            event_vocab_size, event_embed_dim, padding_idx=0
        )
        self.direction_embed = nn.Embedding(
            direction_vocab_size, direction_embed_dim, padding_idx=0
        )
        self.bar_position_embed = nn.Embedding(
            bar_position_vocab_size, bar_position_embed_dim, padding_idx=0
        )

        # === Continuous Projection ===
        self.cont_proj = nn.Linear(1, continuous_embed_dim)

        # === Input Fusion ===
        # Total: level + event + direction + bar_pos + continuous
        total_embed_dim = (
            level_embed_dim
            + event_embed_dim
            + direction_embed_dim
            + bar_position_embed_dim
            + continuous_embed_dim
        )
        self.input_proj = nn.Linear(total_embed_dim, d_model)

        # === Positional Encoding ===
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # === Transformer Encoder ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # === Output Layers ===
        # MLP projection to latent
        self.latent_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, latent_dim),
        )

        # Prediction head
        self.pred_head = nn.Linear(latent_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming initialization."""
        for name, p in self.named_parameters():
            if "embed" in name:
                # Embedding layers - normal initialization
                if p.dim() > 1:
                    nn.init.normal_(p, mean=0.0, std=0.02)
            elif p.dim() > 1:
                # Linear layers - Xavier uniform
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(
        self,
        x_cat: torch.Tensor,
        x_cont: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x_cat: (batch, seq_len, 4) - Categorical IDs [Level, Event, Dir, BarPos]
            x_cont: (batch, seq_len, 1) - Continuous features [PriceDelta]
            mask: (batch, seq_len) - True = valid position, False = padding

        Returns:
            latent: (batch, latent_dim) - Latent representation
            prediction: (batch, 1) - Outcome prediction
        """
        batch_size, seq_len, _ = x_cat.shape

        # === Embed Categoricals ===
        level_emb = self.level_embed(x_cat[:, :, 0])  # (batch, seq, level_dim)
        event_emb = self.event_embed(x_cat[:, :, 1])  # (batch, seq, event_dim)
        direction_emb = self.direction_embed(x_cat[:, :, 2])  # (batch, seq, dir_dim)
        bar_pos_emb = self.bar_position_embed(x_cat[:, :, 3])  # (batch, seq, barpos_dim)

        # === Project Continuous ===
        cont_proj = self.cont_proj(x_cont)  # (batch, seq, continuous_dim)

        # === Concatenate and Project to d_model ===
        fused = torch.cat(
            [level_emb, event_emb, direction_emb, bar_pos_emb, cont_proj], dim=-1
        )
        x = self.input_proj(fused)  # (batch, seq, d_model)

        # === Add Positional Encoding ===
        x = self.pos_encoding(x)

        # === Create Attention Mask ===
        # PyTorch Transformer expects: True = IGNORE position
        if mask is not None:
            attn_mask = ~mask  # Invert: True becomes padding to ignore
        else:
            attn_mask = None

        # === Transformer Encoder ===
        encoded = self.transformer_encoder(x, src_key_padding_mask=attn_mask)

        # === Last Token Pooling ===
        # Get the last valid token for each sequence
        if mask is not None:
            # Find last valid position for each sequence
            lengths = mask.sum(dim=1)  # (batch,)
            batch_indices = torch.arange(batch_size, device=x.device)
            last_indices = torch.clamp(lengths - 1, min=0)  # 0-indexed, clamp for safety
            last_token = encoded[batch_indices, last_indices]  # (batch, d_model)
        else:
            last_token = encoded[:, -1, :]  # (batch, d_model)

        # === Project to Latent ===
        latent = self.latent_proj(last_token)  # (batch, latent_dim)

        # === Prediction Head ===
        prediction = self.pred_head(latent)  # (batch, 1)

        return latent, prediction

    def encode(
        self,
        x_cat: torch.Tensor,
        x_cont: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get latent representation only (for inference/embedding).

        Args:
            x_cat: (batch, seq_len, 4) - Categorical IDs
            x_cont: (batch, seq_len, 1) - Continuous features
            mask: (batch, seq_len) - Attention mask

        Returns:
            latent: (batch, latent_dim) - Latent representation
        """
        latent, _ = self.forward(x_cat, x_cont, mask)
        return latent

    @classmethod
    def from_config(cls, config) -> "TimeStructureModel":
        """
        Create model from TransformerConfig.

        Args:
            config: TransformerConfig instance

        Returns:
            Initialized TimeStructureModel
        """
        return cls(
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout_rate,
            level_vocab_size=config.level_vocab_size,
            event_vocab_size=config.event_vocab_size,
            direction_vocab_size=config.direction_vocab_size,
            bar_position_vocab_size=config.bar_position_vocab_size,
            level_embed_dim=config.level_embed_dim,
            event_embed_dim=config.event_embed_dim,
            direction_embed_dim=config.direction_embed_dim,
            bar_position_embed_dim=config.bar_position_embed_dim,
            continuous_embed_dim=config.continuous_embed_dim,
            latent_dim=config.latent_dim,
            max_seq_len=config.sequence_length + 100,
        )
