import torch
import torch.nn as nn
import torch.nn.functional as F

class IntrinsicCuriosityModule(nn.Module):
    """
    ICM: Predicts next state given current state and action.
    Surprise (Prediction Error) = Intrinsic Reward.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        # Forward Model: (s_t, a_t) -> s_{t+1}
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

        # Inverse Model: (s_t, s_{t+1}) -> a_t
        self.inverse_model = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        self.action_emb = nn.Embedding(action_dim, action_dim)

    def get_intrinsic_reward(self, state, next_state, action_idx):
        a_emb = self.action_emb(action_idx)

        # Forward Prediction
        pred_next_state = self.forward_model(torch.cat([state, a_emb], dim=-1))

        # Error = MSE(Real Next, Pred Next)
        loss = F.mse_loss(pred_next_state, next_state, reduction='none').mean(dim=-1)

        return loss # Intrinsic Reward
