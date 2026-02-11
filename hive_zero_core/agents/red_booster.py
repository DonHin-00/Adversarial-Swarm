"""
Red Team Pre-Attack Booster — Adversarial Payload Hardening

Pre-processes payloads *before* they are sent, adversarially hardening
them against the full blue-team detection stack (WAF, EDR, SIEM, IDS).
Acts as a learned "evasion compiler" that takes a raw payload embedding
and transforms it to minimise the probability of detection by any blue
agent, while preserving the payload's functional semantics.

Architecture
------------
Agent_PreAttackBooster
    1.  Receives the raw payload embedding from PayloadGen / Mutator.
    2.  Runs a lightweight *inner adversarial loop*: for each blue-team
        detector, it computes the gradient of detection probability w.r.t.
        the payload, then steps the payload in the evasion direction.
    3.  Applies a *semantic anchor loss* that keeps the hardened payload
        close to the original in embedding space, preventing semantic drift.
    4.  Outputs the hardened payload embedding ready for delivery.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from hive_zero_core.agents.base_expert import BaseExpert


class Agent_PreAttackBooster(BaseExpert):
    """
    Expert 15: Adversarial Payload Hardener

    Transforms raw payload embeddings to evade an ensemble of blue-team
    detectors via multi-objective gradient descent.  The booster maintains
    its own lightweight *evasion encoder* that learns a general-purpose
    evasion transformation, supplemented by an optional per-payload
    refinement loop at inference time.

    Parameters
    ----------
    observation_dim : int
        Dimensionality of the payload embedding.
    action_dim : int
        Output dimensionality (should match payload dim for pass-through).
    hidden_dim : int
        Width of the evasion encoder.
    refine_steps : int
        Number of gradient-based refinement iterations at inference time.
    refine_lr : float
        Learning rate for the inner refinement loop.
    semantic_weight : float
        Coefficient for the semantic-anchor L2 loss that prevents the
        hardened payload from drifting too far from the original.
    """

    def __init__(self, observation_dim: int, action_dim: int,
                 hidden_dim: int = 128, refine_steps: int = 3,
                 refine_lr: float = 0.01, semantic_weight: float = 0.5):
        super().__init__(observation_dim, action_dim, name="PreAttackBooster",
                         hidden_dim=hidden_dim)

        self.refine_steps = refine_steps
        self.refine_lr = refine_lr
        self.semantic_weight = semantic_weight

        # Evasion encoder: learns a general-purpose evasion transform
        self.evasion_encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Residual gate: blends original payload with evasion transform
        self.gate = nn.Sequential(
            nn.Linear(observation_dim + action_dim, action_dim),
            nn.Sigmoid(),
        )

        # Projection to match action_dim if observation_dim differs
        if observation_dim != action_dim:
            self.residual_proj = nn.Linear(observation_dim, action_dim)
        else:
            self.residual_proj = nn.Identity()

        # Blue-team detector references (populated by HiveMind at init)
        self._blue_detectors: List[nn.Module] = []

    def register_blue_team(self, detectors: List[nn.Module]):
        """
        Register blue-team detector modules for the inner adversarial loop.
        Called by HiveMind after construction.
        """
        self._blue_detectors = detectors

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor],
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Raw payload embedding [batch, observation_dim].

        Returns:
            Hardened payload embedding [batch, action_dim].
        """
        # --- Stage 1: Learned evasion transform ---
        evasion = self.evasion_encoder(x)

        # Gated residual blend
        residual = self.residual_proj(x)
        gate_input = torch.cat([x, evasion], dim=-1)
        g = self.gate(gate_input)
        hardened = g * evasion + (1 - g) * residual

        # --- Stage 2: Optional inner refinement loop ---
        if self._blue_detectors and self.refine_steps > 0:
            hardened = self._refine(hardened, x)

        return hardened

    def _refine(self, payload: torch.Tensor,
                original: torch.Tensor) -> torch.Tensor:
        """
        Gradient-based refinement against registered blue-team detectors.

        For each step:
            1. Compute P(Blocked) from each detector.
            2. Sum detection losses → total_detect.
            3. Add semantic anchor loss: ‖payload − original‖².
            4. Step payload in the negative-gradient direction.

        Dimension matching: blue detectors expect [B, observation_dim]
        inputs, so the payload is projected to observation_dim before
        being passed to each detector.
        """
        refined = payload.clone().detach().requires_grad_(True)
        original_anchor = self.residual_proj(original).detach()

        for _ in range(self.refine_steps):
            if refined.grad is not None:
                refined.grad.zero_()

            total_detect = torch.tensor(0.0, device=refined.device)

            for detector in self._blue_detectors:
                try:
                    # Ensure the payload matches the detector's expected input dim
                    detector_input = self.ensure_dimension(
                        refined, detector.observation_dim
                    )

                    if hasattr(detector, 'model'):
                        logits = detector.model(
                            inputs_embeds=detector_input.unsqueeze(1)
                        ).logits
                    else:
                        logits = detector(detector_input)

                    # P(Blocked) is index 0; maximise P(Allowed) = minimise P(Blocked)
                    if logits.dim() >= 2 and logits.size(-1) >= 2:
                        p_blocked = F.softmax(logits, dim=-1)[:, 0]
                    else:
                        p_blocked = torch.sigmoid(logits).mean(dim=-1)

                    total_detect = total_detect + p_blocked.mean()
                except Exception:
                    continue  # Skip detectors with incompatible shapes

            # Semantic anchor: don't drift too far
            anchor_loss = F.mse_loss(refined, original_anchor)
            loss = total_detect + self.semantic_weight * anchor_loss

            if loss.requires_grad:
                loss.backward()
                with torch.no_grad():
                    if refined.grad is not None:
                        refined = refined - self.refine_lr * refined.grad
                        refined = refined.detach().requires_grad_(True)

        return refined.detach()
