from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv, to_hetero

from hive_zero_core.agents.base_expert import BaseExpert, SkillLevel


class GNNModule(nn.Module):
    def __init__(self, observation_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        # Set add_self_loops=False for heterogeneous graphs
        self.conv1 = GATv2Conv(
            observation_dim, hidden_dim, heads=4, concat=True, add_self_loops=False
        )
        self.conv2 = GATv2Conv(
            hidden_dim * 4, action_dim, heads=1, concat=False, add_self_loops=False
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class CartographerAgent(BaseExpert):
    """
    Expert 1: Temporal Graph Attention Network (T-GAT)
    Utilizes GATv2Conv and GRU to model node history and complex topology.

    Primary Skills: Network Topology Mapping, Graph Analysis
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__(
            observation_dim,
            action_dim,
            name="Cartographer",
            hidden_dim=hidden_dim,
            primary_skills=["recon_001", "recon_002"],  # Network mapping, port scanning
            secondary_skills=["recon_003"],  # Temporal pattern analysis
            skill_level=SkillLevel.EXPERT,
        )

        # GATv2 Architecture
        gnn_base = GNNModule(observation_dim, hidden_dim, action_dim)

        # Convert to Hetero
        self.metadata = (
            ["ip", "port", "protocol"],
            [("ip", "flow", "ip"), ("ip", "binds", "port"), ("port", "uses", "protocol")],
        )
        self.gnn = to_hetero(gnn_base, self.metadata, aggr="sum")

        # Temporal Component
        self.history_gru = nn.GRU(action_dim, action_dim, batch_first=True)

    def _forward_impl(
        self,
        x: Union[torch.Tensor, HeteroData],
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(x, HeteroData):
            # 1. Structural Reasoning
            x_dict = x.x_dict
            edge_index_dict = x.edge_index_dict

            # Apply Hetero GNN
            out_dict = self.gnn(x_dict, edge_index_dict)

            # 2. Temporal Reasoning (Node History)
            if "ip" in out_dict:
                ip_emb = out_dict["ip"]
            else:
                # Ensure the fallback tensor is created on the same device as the model
                try:
                    device = next(self.parameters()).device
                except StopIteration:
                    device = torch.device("cpu")
                ip_emb = torch.zeros(0, self.action_dim, device=device)

            if ip_emb.size(0) > 0:
                # Mock history for prototype (Batch=1, Seq=NodeCount, Dim=ActionDim)
                history = ip_emb.unsqueeze(0)
                h_out, _ = self.history_gru(history)
                return h_out.squeeze(0)

            return ip_emb
        else:
            return torch.zeros(x.size(0), self.action_dim, device=x.device)


class DeepScopeAgent(BaseExpert):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="DeepScope", hidden_dim=hidden_dim)
        self.priority_net = nn.Linear(observation_dim, action_dim)

    def _forward_impl(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.priority_net(x)


class ChronosAgent(BaseExpert):
    """
    Expert 3: Transformer-Enhanced Forecasting
    Uses a Transformer Encoder with causal masking to forecast packet inter-arrival times.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 1000,
    ):
        super().__init__(observation_dim, action_dim, name="Chronos", hidden_dim=hidden_dim)
        self.max_seq_len = max_seq_len

        self.embedding = nn.Linear(1, hidden_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Linear(hidden_dim, action_dim)

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)

    def _forward_impl(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: [batch, seq_len] - raw inter-arrival times
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [batch, seq_len, 1]

        batch_size, seq_len, _ = x.shape

        # Hardening: Check sequence length
        if seq_len > self.max_seq_len:
            self.logger.warning(
                f"Input sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}. Truncating."
            )
            x = x[:, -self.max_seq_len :, :]
            seq_len = self.max_seq_len

        # 1. Embedding + Positional Encoding
        h = self.embedding(x) + self.pos_encoder[:, :seq_len, :]

        # 2. Causal Transformer Encoding
        mask = self._generate_square_subsequent_mask(seq_len).to(x.device)
        feat = self.transformer(h, mask=mask)

        # 3. Forecast Next arrival
        out = self.head(feat[:, -1, :])  # Use last hidden state
        return out
