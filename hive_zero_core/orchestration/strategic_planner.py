import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

class StrategicPlanner(nn.Module):
    """
    Hierarchical Controller (Meta-Level).
    Sets high-level strategic goals (Latent Goals) that condition the HiveMind.
    Goals: 0=Recon, 1=Infiltrate, 2=Persist, 3=Exfiltrate
    """
    def __init__(self, observation_dim: int, goal_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        # Encoder for Global State
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Policy Head (Goal Selection)
        self.policy_head = nn.Linear(hidden_dim, goal_dim)

        # Goal Embedding (to pass to HiveMind)
        self.goal_embedding = nn.Embedding(goal_dim, hidden_dim)

    def forward(self, global_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        # state: [Batch, Obs]
        feat = self.encoder(global_state)
        logits = self.policy_head(feat)
        probs = F.softmax(logits, dim=-1)

        # Sample Goal (Gumbel-Softmax for differentiability if needed, or argmax)
        # For training, we might want gradients.
        if self.training:
            goal_onehot = F.gumbel_softmax(logits, tau=1.0, hard=True)
            goal_idx = torch.argmax(goal_onehot, dim=-1)
        else:
            goal_idx = torch.argmax(probs, dim=-1)

        # Get Goal Embedding
        goal_emb = self.goal_embedding(goal_idx)

        return {"goal_logits": logits, "goal_idx": goal_idx, "goal_embedding": goal_emb}
