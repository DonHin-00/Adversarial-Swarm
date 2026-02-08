import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from typing import Optional, Dict
from hive_zero_core.agents.base_expert import BaseExpert
import ipaddress
import logging

class Agent_Cartographer(BaseExpert):
    """
    Expert 1: Reconnaissance & Mapping (T-GAT)
    Deep Graph Attention Network that handles temporal evolution of the network.
    Predicts hidden links (e.g., pivot opportunities) and node importance (Centrality).
    """
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__(observation_dim, action_dim, name="Cartographer", hidden_dim=hidden_dim)

        # Temporal Component: Process node history with GRU
        self.temporal_encoder = nn.GRU(observation_dim, hidden_dim, batch_first=True)

        # Spatial Component: Multi-Head Graph Attention
        heads = 4
        # Layer 1: Learn local neighborhood features
        self.conv1 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, dropout=0.3, concat=True)
        # Layer 2: Deeper interactions
        self.conv2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, dropout=0.3, concat=True)
        # Layer 3: Output layer (Link Prediction Probability)
        self.conv3 = GATv2Conv(hidden_dim * heads, action_dim, heads=1, concat=False, dropout=0.3)

        # Auxiliary Head: Node Importance (Centrality Prediction)
        self.centrality_head = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Context is expected to be the Edge Index [2, num_edges]
        if context is None:
             # Fallback
             return torch.zeros((x.size(0), self.action_dim), device=x.device)

        edge_index = context

        # 1. Temporal Encoding (Simulated)
        # If x is [nodes, features], we treat it as current snapshot.
        # Ideally, x would be [nodes, seq_len, features].
        if x.dim() == 2:
            x_seq = x.unsqueeze(1) # Add seq dim [nodes, 1, features]
        else:
            x_seq = x

        # GRU expects [batch, seq, features]
        # Project features first if needed? No, input dim matches obs dim.
        out, hn = self.temporal_encoder(x_seq)
        x_proj = hn[-1] # [nodes, hidden_dim]

        # 2. Spatial Graph Attention
        x1 = F.dropout(x_proj, p=0.3, training=self.training)
        x1 = self.conv1(x1, edge_index)
        x1 = F.elu(x1)

        x2 = F.dropout(x1, p=0.3, training=self.training)
        x2 = self.conv2(x2, edge_index)
        x2 = F.elu(x2)

        # 3. Task Heads
        # Link Prediction Embeddings
        node_embeddings = self.conv3(x2, edge_index) # [nodes, action_dim]

        # Return embeddings (Centrality is auxiliary, handled separately or appended?)
        # BaseExpert expects single tensor return.
        # Let's return embeddings for now.
        return node_embeddings

class Agent_DeepScope(BaseExpert):
    """
    Expert 2: Constraint Masking (Logic-Based + Learned)
    Parses Rules of Engagement (RoE) and applies strict masking.
    Learns to prioritize valid targets within the constraints.
    """
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="DeepScope", hidden_dim=hidden_dim)

        # Learned Component: Prioritize valid targets based on observation
        self.priority_net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim) # Raw priority scores
        )

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: Global/Node features

        # 1. Compute Priority Scores
        priority_logits = self.priority_net(x)

        # 2. Apply Hard Constraints
        if mask is not None:
            # External mask provided
            final_mask = mask
        else:
            # Fallback: All blocked (Safe default)
            final_mask = torch.zeros_like(priority_logits)

        # Ensure broadcast
        if final_mask.shape != priority_logits.shape:
            # Attempt basic broadcast logic
            if final_mask.size(0) == priority_logits.size(0) and final_mask.dim() == 1:
                 final_mask = final_mask.unsqueeze(-1)
            elif final_mask.dim() == 1 and final_mask.size(0) == priority_logits.size(1):
                 # Mask applies to action dim globally
                 final_mask = final_mask.unsqueeze(0)

        # 3. Masking Logic
        # Valid actions get priority_logits. Invalid get -1e9.
        masked_output = priority_logits * final_mask + (1.0 - final_mask) * -1e9

        return masked_output

class Agent_Chronos(BaseExpert):
    """
    Expert 3: Time-Series / Heartbeat (Transformer)
    Uses Self-Attention to model long-range dependencies in packet flows.
    Detects jitter patterns and predicts safe injection windows.
    """
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64): # Adjusted hidden to match model_dim requirements
        super().__init__(observation_dim, action_dim, name="Chronos", hidden_dim=hidden_dim)

        # Transformer Encoder for Time Series
        # d_model must match hidden_dim
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # Projection from input dim (1 for just dt) to hidden_dim
        self.input_proj = nn.Linear(1, hidden_dim)

        # Positional Encoding (Learnable)
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, hidden_dim)) # Max seq len 100

        # Heads
        self.next_arrival_head = nn.Linear(hidden_dim, action_dim) # Predict next dt

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [batch, seq_len] or [batch, seq_len, 1]
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        batch_size, seq_len, feat_dim = x.size()

        # 1. Project Input
        x_emb = self.input_proj(x) # [batch, seq_len, hidden]

        # 2. Add Positional Encoding
        # Slice to current seq len. Clamp if seq_len > 100
        if seq_len > 100:
             pos = self.pos_encoder[:, :, :] # Just use full enc? Or repeat?
             # Truncate input for prototype safety
             x_emb = x_emb[:, :100, :]
             seq_len = 100
        else:
             pos = self.pos_encoder[:, :seq_len, :]

        x_emb = x_emb + pos

        # 3. Transformer Pass
        # Causal Mask? For forecasting, we mask future.
        # Generate square subsequent mask
        # src_mask argument in transformer is (S, S) or (N*num_heads, S, S)
        # Using bool mask for clarity: True means ignore (masked).
        # nn.TransformerEncoder takes mask where True positions are not allowed to attend.
        # Wait, PyTorch docs say: "If a BoolTensor is provided, positions with True are not allowed to attend while False values will be unchanged."
        # Standard causal mask is upper triangular.

        # Using float mask: -inf for masked, 0 for unmasked.
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)

        x_trans = self.transformer(x_emb, mask=causal_mask, is_causal=True)

        # 4. Predict from last state
        last_state = x_trans[:, -1, :] # [batch, hidden]

        predicted_dt = F.softplus(self.next_arrival_head(last_state)) # Must be positive

        return predicted_dt
