import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, to_hetero
from torch_geometric.data import HeteroData
from typing import Optional, Dict, Union
from hive_zero_core.agents.base_expert import BaseExpert

class GNNModule(nn.Module):
    def __init__(self, observation_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.conv1 = GATv2Conv(observation_dim, hidden_dim, heads=4, concat=True)
        self.conv2 = GATv2Conv(hidden_dim * 4, action_dim, heads=1, concat=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class Agent_Cartographer(BaseExpert):
    """
    Expert 1: Temporal Graph Attention Network (T-GAT)
    Utilizes GATv2Conv and GRU to model node history and complex topology.
    """
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="Cartographer", hidden_dim=hidden_dim)

        # GATv2 Architecture
        gnn_base = GNNModule(observation_dim, hidden_dim, action_dim)

        # Convert to Hetero
        self.metadata = (
            ['ip', 'port', 'protocol'],
            [('ip', 'flow', 'ip'), ('ip', 'binds', 'port'), ('port', 'uses', 'protocol')]
        )
        self.gnn = to_hetero(gnn_base, self.metadata, aggr='sum')

        # Temporal Component
        self.history_gru = nn.GRU(action_dim, action_dim, batch_first=True)

    def _forward_impl(self, x: Union[torch.Tensor, HeteroData], context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if isinstance(x, HeteroData):
            # 1. Structural Reasoning
            x_dict = x.x_dict
            edge_index_dict = x.edge_index_dict

            # Apply Hetero GNN
            out_dict = self.gnn(x_dict, edge_index_dict)

            # 2. Temporal Reasoning (Node History)
            ip_emb = out_dict.get('ip', torch.zeros(0, self.action_dim))

            if ip_emb.size(0) > 0:
                # Mock history for prototype (Batch=1, Seq=NodeCount, Dim=ActionDim)
                history = ip_emb.unsqueeze(0)
                h_out, _ = self.history_gru(history)
                return h_out.squeeze(0)

            return ip_emb
        else:
            return torch.zeros(x.size(0), self.action_dim, device=x.device)

class Agent_DeepScope(BaseExpert):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="DeepScope", hidden_dim=hidden_dim)
        self.priority_net = nn.Linear(observation_dim, action_dim)

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.priority_net(x)

class Agent_Chronos(BaseExpert):
    """
    Expert 3: Transformer-Enhanced Forecasting
    Uses a Transformer Encoder with causal masking to forecast packet inter-arrival times.
    """
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__(observation_dim, action_dim, name="Chronos", hidden_dim=hidden_dim)

        self.embedding = nn.Linear(1, hidden_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, 1000, hidden_dim)) # Max seq length 1000

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Linear(hidden_dim, action_dim)

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [batch, seq_len] - raw inter-arrival times
        if x.dim() == 2:
            x = x.unsqueeze(-1) # [batch, seq_len, 1]

        batch_size, seq_len, _ = x.shape

        # 1. Embedding + Positional Encoding
        h = self.embedding(x) + self.pos_encoder[:, :seq_len, :]

        # 2. Causal Transformer Encoding
        mask = self._generate_square_subsequent_mask(seq_len).to(x.device)
        feat = self.transformer(h, mask=mask)

        # 3. Forecast Next arrival
        out = self.head(feat[:, -1, :]) # Use last hidden state
        return out
