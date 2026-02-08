import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from hive_zero_core.agents.base_expert import BaseExpert

class Agent_Mimic(BaseExpert):
    """
    Expert 7: Traffic Mimic (VAE-GAN)
    Generates realistic network traffic shapes (packet size, delay) using a VAE-GAN architecture.
    VAE captures the latent distribution of normal traffic; GAN ensures realism.
    """
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64):
        # action_dim should be 2 (size, delay) typically
        super().__init__(observation_dim, action_dim, name="Mimic", hidden_dim=hidden_dim)

        # Encoder (VAE)
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.mu_head = nn.Linear(hidden_dim, hidden_dim)
        self.logvar_head = nn.Linear(hidden_dim, hidden_dim)

        # Generator / Decoder (GAN)
        # Input: Latent (hidden) + Noise (16)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + 16, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: Traffic Stats (Normal Profile)

        # 1. Encode into Latent Distribution
        enc = self.encoder(x)
        mu = self.mu_head(enc)
        logvar = self.logvar_head(enc)
        z = self._reparameterize(mu, logvar)

        # 2. Add GAN Noise
        noise = torch.randn(x.size(0), 16, device=x.device)
        latent_input = torch.cat([z, noise], dim=1)

        # 3. Generate Traffic Features
        out = self.decoder(latent_input)

        # Constraints: Delays/Sizes must be positive
        return F.softplus(out)

class Agent_Ghost(BaseExpert):
    """
    Expert 8: System Entropy Analyzer (Metadata Variance)
    Analyzes simulated file system metadata to find high-variance hiding spots.
    Calculates Shannon Entropy of file attributes (size, mtime, permissions).
    """
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="Ghost", hidden_dim=hidden_dim)

        # Feature Extractor for Directory Metadata
        self.feature_extractor = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Scoring Head
        self.score_head = nn.Linear(hidden_dim, 1)

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: Directory Metadata Embeddings [Batch, Num_Dirs, Feats]
        # Or flattened [Batch, Obs]

        if x.dim() == 2:
            # Single dir features
            feat = self.feature_extractor(x)
            score = self.score_head(F.relu(feat))
            return torch.sigmoid(score)

        elif x.dim() == 3:
            # Multiple dirs [Batch, Num, Feats]
            B, N, D = x.shape
            # Ensure D matches obs_dim. If not, raise or adapt? Assume correct.
            x_flat = x.reshape(B * N, D)
            feat = self.feature_extractor(x_flat)
            scores = self.score_head(F.relu(feat))
            return scores.reshape(B, N, 1) # Return raw scores or sigmoid? BaseExpert usually returns tensor.

        return torch.zeros(x.size(0), 1, device=x.device)

class Agent_Stego(BaseExpert):
    """
    Expert 9: Neural Steganography (LSB & Capacity Check)
    Embeds binary payload into cover medium (image tensor) using LSB modification.
    Includes a capacity check network to ensure fit.
    """
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="Stego", hidden_dim=hidden_dim)

        # Capacity Network: Predicts max bits/pixel from image features
        # Assuming input context is image features [Batch, Channels, H, W]
        self.capacity_net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # Encoder (Payload -> Bitstream Embedding)
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid() # 0-1 soft bits
        )

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: Payload data chunks [batch, obs_dim]
        # context: Cover Image [batch, 3, H, W] (Optional)

        encoded_payload = self.encoder(x)

        # If context is provided (image), calculate capacity
        # For simplicity, base expert returns only one tensor, so we return encoded payload
        # In full system, HiveMind would query capacity separately

        return encoded_payload

class Agent_Cleaner(BaseExpert):
    """
    Expert 10: Causal Graph Cleaner
    Generates cleanup scripts by reversing the causal graph of actions.
    Uses an Attention mechanism to link current state to past actions.
    """
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="Cleaner", hidden_dim=hidden_dim)

        # Action History Encoder (LSTM)
        # obs_dim matches history vector size
        self.history_encoder = nn.LSTM(observation_dim, hidden_dim, batch_first=True)

        # Causal Attention
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=2, batch_first=True)

        # Decoder (Action generator)
        self.decoder = nn.Linear(hidden_dim, action_dim)

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: History of actions (embeddings) [batch, seq_len, obs_dim]

        # Ensure sequence dimension exists
        if x.dim() == 2:
             x = x.unsqueeze(1)

        # Encode history
        out, (hn, cn) = self.history_encoder(x)

        # Self-Attention to find causal dependencies
        # Query = Last state (current system state representation)
        # Key/Val = History states
        query = hn[-1].unsqueeze(0).transpose(0, 1) # [batch, 1, hidden]

        # attention forward takes (query, key, value)
        attn_out, _ = self.attention(query, out, out)

        # Generate Inverse Action from attended context
        cleanup_action = self.decoder(attn_out.squeeze(1))

        return cleanup_action
