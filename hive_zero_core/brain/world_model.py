import torch
import torch.nn as nn
from typing import Tuple

class LatentWorldModel(nn.Module):
    """
    Dreamer-Lite: Predicts latent transitions and rewards.
    Allows planning in latent space without executing real actions.
    """
    def __init__(self, observation_dim: int, action_dim: int, latent_dim: int = 64):
        super().__init__()
        # Encoder: Obs -> Latent State
        self.encoder = nn.Linear(observation_dim, latent_dim)

        # Transition Model: (State, Action) -> Next State (Latent)
        self.transition = nn.Sequential(
            nn.Linear(latent_dim + action_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        # Reward Model: (State, Action) -> Reward
        self.reward_predictor = nn.Sequential(
            nn.Linear(latent_dim + action_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, 1)
        )

        # Action Embedding (if discrete experts)
        self.action_emb = nn.Embedding(action_dim, action_dim)

    def forward(self, obs: torch.Tensor, action_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # action_idx: [Batch]
        z = self.encoder(obs)
        a = self.action_emb(action_idx)

        za = torch.cat([z, a], dim=-1)

        next_z = self.transition(za)
        pred_reward = self.reward_predictor(za)

        return next_z, pred_reward

    def dream(self, start_obs: torch.Tensor, policy_net: nn.Module, horizon: int = 5):
        """
        Simulate a trajectory in latent space.
        """
        self.encoder(start_obs)

        for _ in range(horizon):
            pass
