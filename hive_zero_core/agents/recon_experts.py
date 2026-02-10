from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from hive_zero_core.agents.base_expert import BaseExpert


class Agent_Cartographer(BaseExpert):
    """
    Expert 1: Reconnaissance & Mapping (GAT)
    Predicts hidden links in the network.
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64):
        # Action dim here is effectively num_nodes (adjacency probability) or embedding dim
        # Assuming output is updated node embeddings for link prediction
        super().__init__(observation_dim, action_dim, name="Cartographer", hidden_dim=hidden_dim)

        self.conv1 = GATv2Conv(observation_dim, hidden_dim, heads=4, dropout=0.2)
        # Output dim matches action_dim for compatibility
        self.conv2 = GATv2Conv(hidden_dim * 4, action_dim, heads=1, concat=False, dropout=0.2)

    def _forward_impl(
        self, x: torch.Tensor, context: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Context is expected to be the Edge Index from the Graph
        # x is Node Features [num_nodes, features]
        if context is None:
            # Fallback if no graph structure provided
            return torch.zeros((x.size(0), self.action_dim), device=x.device)

        edge_index = context

        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)

        return x


class Agent_DeepScope(BaseExpert):
    """
    Expert 2: Constraint Masking
    Applies RoE masks to action logits.
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="DeepScope", hidden_dim=hidden_dim)
        # DeepScope includes a learnable layer to transform observations to logits before masking
        self.adapter = nn.Linear(observation_dim, action_dim)

    def _forward_impl(
        self, x: torch.Tensor, context: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        logits = self.adapter(x)

        if mask is not None:
            # Hard Masking: -1e9 for invalid actions
            # Ensure mask matches logits shape or broadcasts
            if mask.shape != logits.shape:
                # Simple check, real implementation would handle broadcasting carefully
                # If logits is [batch, action_dim] and mask is [batch, action_dim] -> OK
                # If mask is [action_dim] -> OK
                pass

            # (1 - mask) * large_negative + mask * logits
            # Assuming mask is 1 for valid, 0 for invalid
            # Ensure 1-mask is broadcastable
            masked_logits = logits * mask + (1 - mask) * -1e9
            return masked_logits

        return logits


class Agent_Chronos(BaseExpert):
    """
    Expert 3: Time-Series / Heartbeat
    Predicts optimal injection timestamp.
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="Chronos", hidden_dim=hidden_dim)

        # Input: Sequence of inter-arrival times [batch, seq_len, 1]
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.head = nn.Linear(
            hidden_dim, action_dim
        )  # Output: timestamp scalar or distribution parameters

    def _forward_impl(
        self, x: torch.Tensor, context: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Hardening: Check input shape. Expecting [batch, seq_len] or [batch, seq_len, 1]
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [batch, seq_len] -> [batch, seq_len, 1]

        # Z-score normalization for outlier hardening (simplified per-batch)
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-6
        x_norm = (x - mean) / std

        out, (hn, cn) = self.lstm(x_norm)

        # Take last hidden state
        last_hidden = out[:, -1, :]
        prediction = self.head(last_hidden)

        return prediction
