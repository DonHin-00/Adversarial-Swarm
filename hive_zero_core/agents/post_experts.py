import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from hive_zero_core.agents.base_expert import BaseExpert


class SpectralNormLinear(nn.Module):
    """Linear layer with spectral normalisation for Lipschitz-constrained mappings."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.utils.parametrizations.spectral_norm(
            nn.Linear(in_features, out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class Agent_Mimic(BaseExpert):
    """
    Expert 7: Spectrally-Normalised Flow-GAN Generator

    Generates network traffic shapes (packet size, inter-arrival delay) that
    mimic a target baseline distribution.  Spectral normalisation on all
    linear layers stabilises GAN training and prevents mode collapse.
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64,
                 noise_dim: int = 32):
        super().__init__(observation_dim, action_dim, name="Mimic", hidden_dim=hidden_dim)
        self.noise_dim = noise_dim

        self.fc1 = SpectralNormLinear(observation_dim + noise_dim, hidden_dim)
        self.fc2 = SpectralNormLinear(hidden_dim, hidden_dim)
        self.fc3 = SpectralNormLinear(hidden_dim, action_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor],
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.size(0)
        noise = torch.randn(batch_size, self.noise_dim, device=x.device)
        inp = torch.cat([x, noise], dim=1)

        out = F.gelu(self.norm1(self.fc1(inp)))
        out = F.gelu(self.norm2(self.fc2(out)))
        out = self.fc3(out)

        # Packet sizes / delays must be positive
        return F.softplus(out)


class Agent_Ghost(BaseExpert):
    """
    Expert 8: Entropy-Aware System Concealment

    Scores candidate hiding locations (directories, processes) by combining
    a learned feature extractor with an explicit entropy gate. LayerNorm and
    a residual connection improve gradient flow and training stability.
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="Ghost", hidden_dim=hidden_dim)

        self.feature_net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.score_head = nn.Linear(hidden_dim, action_dim)

        # Entropy gate: learns to suppress high-variance (conspicuous) locations
        self.entropy_gate = nn.Sequential(
            nn.Linear(observation_dim, action_dim),
            nn.Sigmoid(),
        )

        # Residual projection
        self.res_proj = nn.Linear(observation_dim, action_dim)

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor],
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        features = self.feature_net(x)
        scores = self.score_head(features)

        # Gate: attenuate conspicuous candidates
        gate = self.entropy_gate(x)
        gated = scores * gate

        # Residual skip
        return gated + self.res_proj(x)


class Agent_Stego(BaseExpert):
    """
    Expert 9: Variational Autoencoder Steganography

    Encodes binary payloads into a covert-channel representation via a
    variational bottleneck. The KL term regularises the latent space,
    making the encoded signal harder to distinguish from noise.
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64,
                 latent_dim: int = 16):
        super().__init__(observation_dim, action_dim, name="Stego", hidden_dim=hidden_dim)
        self.latent_dim = latent_dim

        # Encoder → μ, log σ²
        self.enc_shared = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        # Decoder: latent → covert channel bits
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid(),  # Normalise to [0, 1] for bit-level embedding
        )

    def _reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor],
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.enc_shared(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)

        z = self._reparameterise(mu, logvar)
        return self.decoder(z)

    def kl_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence for the most recent encoding (training helper)."""
        h = self.enc_shared(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


class Agent_Cleaner(BaseExpert):
    """
    Expert 10: Residual Causal Inference Module

    Computes inverse (cleanup) actions from a history-log embedding using a
    deeper residual MLP with LayerNorm for stable gradient propagation.
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="Cleaner", hidden_dim=hidden_dim)

        self.block1 = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden_dim, action_dim)

        # Residual skip
        self.res_proj = nn.Linear(observation_dim, hidden_dim)

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor],
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.block1(x)
        h = self.block2(h) + self.res_proj(x)  # Residual
        return self.head(h)
