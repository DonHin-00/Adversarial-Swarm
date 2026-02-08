
import torch
from torch import nn


class XAIExplainer:
    """
    Explainability Module.
    Uses Gradient-based attribution to explain gating decisions.
    """
    def __init__(self, gating_network: nn.Module):
        self.gating_network = gating_network

    def explain_decision(self, state: torch.Tensor, target_expert_idx: int) -> torch.Tensor:
        """
        Returns feature attribution map for the selected expert.
        """
        state.requires_grad_(True)

        weights, _ = self.gating_network(state, training=False)

        # Target score: Weight of the specific expert
        score = weights[:, target_expert_idx].sum()

        # Compute Gradients w.r.t Input State
        grads = torch.autograd.grad(score, state)[0]

        # Saliency: |Grads|
        return grads.abs()
