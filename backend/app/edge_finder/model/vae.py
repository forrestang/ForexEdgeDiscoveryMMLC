"""
Variational Autoencoder for waveform matrices.

LSTM-based VAE that encodes variable-length waveform sequences
into fixed-size latent vectors for pattern matching.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Optional


class WaveformEncoder(nn.Module):
    """
    LSTM-based encoder for waveform sequences.

    Takes variable-length [batch, seq_len, 20] inputs and produces
    fixed-size latent parameters (mu, logvar).
    """

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Latent space projections
        lstm_output_dim = hidden_dim * self.num_directions
        self.fc_mu = nn.Linear(lstm_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(lstm_output_dim, latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode sequences to latent parameters.

        Args:
            x: [batch, seq_len, input_dim] padded input
            lengths: [batch] original sequence lengths

        Returns:
            mu: [batch, latent_dim] mean of latent distribution
            logvar: [batch, latent_dim] log variance of latent distribution
        """
        batch_size = x.shape[0]

        # Project input
        x = self.input_proj(x)

        # Pack for efficient LSTM processing
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=True
        )

        # Run LSTM
        _, (hidden, _) = self.lstm(packed)

        # hidden: [num_layers * num_directions, batch, hidden_dim]
        # Take the last layer's hidden states
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]

        # Project to latent space
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)

        return mu, logvar


class WaveformDecoder(nn.Module):
    """
    LSTM-based decoder for waveform sequences.

    Takes latent vectors and target lengths, outputs reconstructed sequences.
    """

    def __init__(
        self,
        output_dim: int = 20,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # Latent to initial hidden state
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.latent_to_cell = nn.Linear(latent_dim, hidden_dim * num_layers)

        # LSTM decoder
        self.lstm = nn.LSTM(
            input_size=output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        z: torch.Tensor,
        target_lengths: torch.Tensor,
        max_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Decode latent vectors to sequences.

        Args:
            z: [batch, latent_dim] latent vectors
            target_lengths: [batch] target sequence lengths
            max_len: Maximum length to decode (defaults to max of target_lengths)

        Returns:
            output: [batch, max_len, output_dim] reconstructed sequences
        """
        batch_size = z.shape[0]
        device = z.device

        if max_len is None:
            max_len = target_lengths.max().item()

        # Initialize hidden states from latent
        hidden = self.latent_to_hidden(z)
        hidden = hidden.view(batch_size, self.num_layers, self.hidden_dim)
        hidden = hidden.permute(1, 0, 2).contiguous()

        cell = self.latent_to_cell(z)
        cell = cell.view(batch_size, self.num_layers, self.hidden_dim)
        cell = cell.permute(1, 0, 2).contiguous()

        # Start with zeros
        decoder_input = torch.zeros(batch_size, 1, self.output_dim, device=device)

        outputs = []
        for t in range(max_len):
            output, (hidden, cell) = self.lstm(decoder_input, (hidden, cell))
            output = self.output_proj(output)
            outputs.append(output)
            decoder_input = output

        # Stack outputs
        output = torch.cat(outputs, dim=1)

        return output


class WaveformVAE(nn.Module):
    """
    Variational Autoencoder for waveform matrices.

    Combines LSTM encoder and decoder with reparameterization trick.
    """

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.latent_dim = latent_dim

        self.encoder = WaveformEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )

        self.decoder = WaveformDecoder(
            output_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reparameterization trick for sampling from latent distribution.

        Args:
            mu: [batch, latent_dim] mean
            logvar: [batch, latent_dim] log variance

        Returns:
            z: [batch, latent_dim] sampled latent vectors
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # During inference, just use the mean
            return mu

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.

        Args:
            x: [batch, seq_len, input_dim] padded input
            lengths: [batch] original sequence lengths

        Returns:
            recon: [batch, seq_len, input_dim] reconstructed sequences
            mu: [batch, latent_dim] latent mean
            logvar: [batch, latent_dim] latent log variance
        """
        # Encode
        mu, logvar = self.encoder(x, lengths)

        # Sample latent
        z = self.reparameterize(mu, logvar)

        # Decode
        max_len = x.shape[1]
        recon = self.decoder(z, lengths, max_len)

        return recon, mu, logvar

    def encode(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode sequences to latent vectors (for inference).

        Args:
            x: [batch, seq_len, input_dim] padded input
            lengths: [batch] original sequence lengths

        Returns:
            z: [batch, latent_dim] latent vectors (mean, no sampling)
        """
        mu, _ = self.encoder(x, lengths)
        return mu

    def decode(
        self,
        z: torch.Tensor,
        target_length: int,
    ) -> torch.Tensor:
        """
        Decode latent vectors to sequences.

        Args:
            z: [batch, latent_dim] latent vectors
            target_length: Length of sequences to generate

        Returns:
            output: [batch, target_length, output_dim] generated sequences
        """
        batch_size = z.shape[0]
        lengths = torch.full((batch_size,), target_length, device=z.device)
        return self.decoder(z, lengths, target_length)


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    mask: torch.Tensor,
    kl_weight: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    Compute VAE loss: reconstruction + KL divergence.

    Args:
        recon: [batch, seq_len, dim] reconstructed sequences
        target: [batch, seq_len, dim] target sequences
        mu: [batch, latent_dim] latent mean
        logvar: [batch, latent_dim] latent log variance
        mask: [batch, seq_len] boolean mask (True = valid position)
        kl_weight: Weight for KL divergence term (beta in beta-VAE)

    Returns:
        total_loss: Scalar loss tensor
        metrics: Dict with individual loss components
    """
    batch_size = recon.shape[0]

    # Masked reconstruction loss (MSE)
    # Time mask: [batch, seq_len, 1] - masks padding
    time_mask = mask.unsqueeze(-1).float()

    # Feature mask: [batch, seq_len, 20] - masks inactive wave levels (zeros)
    # Only compute loss on non-zero targets to avoid gradient distortion from sparse zeros
    feature_mask = (target != 0).float()

    # Combined mask: valid time positions AND non-zero features
    combined_mask = time_mask * feature_mask

    # Compute squared error only at valid, non-zero positions
    sq_error = (recon - target) ** 2
    masked_sq_error = sq_error * combined_mask

    # Mean over all valid non-zero positions
    recon_loss = masked_sq_error.sum() / (combined_mask.sum() + 1e-8)  # epsilon for stability

    # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / batch_size  # Per-sample average

    # Total loss
    total_loss = recon_loss + kl_weight * kl_loss

    metrics = {
        "loss": total_loss.item(),
        "recon_loss": recon_loss.item(),
        "kl_loss": kl_loss.item(),
    }

    return total_loss, metrics
