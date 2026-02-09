import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from typing import Optional, Dict
from hive_zero_core.agents.base_expert import BaseExpert

class Agent_Cartographer(BaseExpert):
    """
    Expert 1: Reconnaissance & Mapping (Deep GAT with Residuals)

    Uses a 3-layer Graph Attention Network (GATv2) with residual connections
    and layer normalisation to produce node embeddings suitable for
    downstream link-prediction. Deeper architecture captures higher-order
    neighbourhood structure than the original 2-layer version.
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="Cartographer", hidden_dim=hidden_dim)

        heads = 4
        self.conv1 = GATv2Conv(observation_dim, hidden_dim, heads=heads, dropout=0.2)
        self.norm1 = nn.LayerNorm(hidden_dim * heads)

        self.conv2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, dropout=0.2)
        self.norm2 = nn.LayerNorm(hidden_dim * heads)

        self.conv3 = GATv2Conv(hidden_dim * heads, action_dim, heads=1, concat=False, dropout=0.2)
        self.norm3 = nn.LayerNorm(action_dim)

        # Residual projection: match dims between conv layers for skip connections
        self.res_proj1 = nn.Linear(observation_dim, hidden_dim * heads)
        self.res_proj2 = nn.Linear(hidden_dim * heads, action_dim)

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor],
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if context is None:
            return torch.zeros((x.size(0), self.action_dim), device=x.device)

        edge_index = context
        identity = x

        # Layer 1 + residual
        h = F.dropout(x, p=0.2, training=self.training)
        h = self.conv1(h, edge_index)
        h = self.norm1(h)
        h = F.elu(h) + self.res_proj1(identity)

        # Layer 2 + residual
        identity2 = h
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.conv2(h, edge_index)
        h = self.norm2(h)
        h = F.elu(h) + identity2

        # Layer 3 + residual
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.conv3(h, edge_index)
        h = self.norm3(h)
        h = h + self.res_proj2(identity2)

        return h


class Agent_DeepScope(BaseExpert):
    """
    Expert 2: Multi-Head Attention Constraint Masking

    Transforms observations into action logits through a two-layer MLP with
    multi-head self-attention, then applies hard Rules-of-Engagement (RoE)
    masks to suppress disallowed actions. Upgraded from a single linear
    adapter to capture richer observation→constraint mappings.
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64,
                 num_heads: int = 4):
        super().__init__(observation_dim, action_dim, name="DeepScope", hidden_dim=hidden_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=observation_dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(observation_dim)
        self.adapter = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor],
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention over the observation (treat dim as seq of length 1)
        if x.dim() == 2:
            x_seq = x.unsqueeze(1)  # [B, 1, D]
        else:
            x_seq = x

        attn_out, _ = self.attention(x_seq, x_seq, x_seq)
        attn_out = self.norm(attn_out + x_seq)  # Residual + LayerNorm

        # Pool back to [B, D] and project to action logits
        pooled = attn_out.mean(dim=1)
        logits = self.adapter(pooled)

        if mask is not None:
            # Broadcast mask to match logits shape
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            if mask.shape[0] == 1 and logits.shape[0] > 1:
                mask = mask.expand(logits.shape[0], -1)
            elif mask.shape != logits.shape:
                # Safety fallback: disable masking if shapes are incompatible
                self.logger.warning(
                    f"Mask shape {mask.shape} incompatible with logits {logits.shape}; "
                    "masking disabled for this forward pass"
                )
                return logits

            masked_logits = logits * mask + (1 - mask) * -1e9
            return masked_logits

        return logits


class Agent_Chronos(BaseExpert):
    """
    Expert 3: Transformer-Based Temporal Encoder

    Replaces the original 2-layer LSTM with a lightweight Transformer encoder
    followed by a learned temporal pooling head. Captures long-range timing
    dependencies more effectively and parallelises across the sequence.
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64,
                 nhead: int = 4, num_layers: int = 2):
        super().__init__(observation_dim, action_dim, name="Chronos", hidden_dim=hidden_dim)

        # Project scalar inter-arrival times to hidden_dim
        self.input_proj = nn.Linear(1, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 4,
            dropout=0.1, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

        # Temporal pooling: learned query that attends to the full sequence
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=nhead, batch_first=True
        )

        self.head = nn.Linear(hidden_dim, action_dim)

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor],
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Expecting [batch, seq_len] or [batch, seq_len, 1]
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [B, S] → [B, S, 1]

        batch_size, seq_len, _ = x.shape

        # Z-score normalisation (robust to outliers)
        mean = x.mean(dim=1, keepdim=True)
        std = torch.clamp(x.std(dim=1, keepdim=True), min=1e-6)
        x_norm = (x - mean) / std

        # Project and add positional encoding
        h = self.input_proj(x_norm)  # [B, S, H]
        h = h + self.pos_encoding[:, :seq_len, :]

        # Transformer encode
        h = self.encoder(h)
        h = self.norm(h)

        # Learned temporal pooling
        query = self.pool_query.expand(batch_size, -1, -1)  # [B, 1, H]
        pooled, _ = self.pool_attn(query, h, h)  # [B, 1, H]
        pooled = pooled.squeeze(1)  # [B, H]

        return self.head(pooled)
