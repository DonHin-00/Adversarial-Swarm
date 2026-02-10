from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hive_zero_core.agents.base_expert import BaseExpert


class Agent_Mimic(BaseExpert):
    """
    Expert 7: Flow-GAN Generator
    Generates network traffic shapes (packet size, delay) to mimic target distribution.
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64):
        # Action dim = 2 (size, delay) typically, or seq_len * 2
        super().__init__(observation_dim, action_dim, name="Mimic", hidden_dim=hidden_dim)

        # Simple GAN Generator Architecture
        # Input: obs (content stats) + noise
        self.fc1 = nn.Linear(observation_dim + 16, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def _forward_impl(
        self, x: torch.Tensor, context: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # x: Input features (stolen data stats + baseline)
        batch_size = x.size(0)

        # Generate noise
        noise = torch.randn(batch_size, 16, device=x.device)

        # Concatenate
        inp = torch.cat([x, noise], dim=1)

        out = F.relu(self.fc1(inp))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        # Output: Packet sizes/delays.
        # Delays/Sizes must be positive -> Softplus
        return F.softplus(out)


class Agent_Ghost(BaseExpert):
    """
    Expert 8: System Entropy Minimizer
    Identifies high variance directories for hiding files.
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="Ghost", hidden_dim=hidden_dim)

        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),  # Score per directory
        )

    def _forward_impl(
        self, x: torch.Tensor, context: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # x: File system metadata stats
        scores = self.net(x)
        return scores


class Agent_Stego(BaseExpert):
    """
    Expert 9: Neural Autoencoder
    Encodes binary data into LSBs or covert channels.
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="Stego", hidden_dim=hidden_dim)

        # Encoder part only for the agent action
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid(),  # Normalize to 0-1 for bit embedding representation
        )

    def _forward_impl(
        self, x: torch.Tensor, context: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # x: Binary data chunks to hide
        encoded = self.encoder(x)
        return encoded


class Agent_Cleaner(BaseExpert):
    """
    Expert 10: Causal Inference Module
    Calculates inverse operations to restore state.
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="Cleaner", hidden_dim=hidden_dim)

        self.inverse_model = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),  # Cleanup script/actions
        )

    def _forward_impl(
        self, x: torch.Tensor, context: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # x: History log embedding
        cleanup_actions = self.inverse_model(x)
        return cleanup_actions
