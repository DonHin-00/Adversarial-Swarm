import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Tuple

class PPOActorCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # Actor Head (Policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic Head (Value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.encoder(state)

        probs = self.actor(feat)
        value = self.critic(feat)

        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob, value

    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.encoder(state)
        probs = self.actor(feat)
        value = self.critic(feat)

        dist = Categorical(probs)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return log_prob, value, entropy
