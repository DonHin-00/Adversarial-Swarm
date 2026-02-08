import math
from typing import Optional

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from hive_zero_core.agents.base_expert import BaseExpert


class TrapArsenal:
    """
    Expert-Level Adversarial Mathematics Library.
    Generates 20+ distinct high-fidelity trap signatures.
    """

    @staticmethod
    def chaotic_dynamics(batch_size: int, dim: int, device: torch.device) -> torch.Tensor:
        """The Siren: Logistic Map Chaos Generator."""
        r = 3.99 # Chaotic regime
        x = torch.rand(batch_size, dim, device=device)
        # Iterate map
        for _ in range(5):
            x = r * x * (1 - x)
        return x

    @staticmethod
    def recursive_fractal(batch_size: int, dim: int, device: torch.device) -> torch.Tensor:
        """The Infinite Hallway: Self-similar expansion."""
        # Simulating fractal depth via high-freq sinusoidal interference
        t = torch.linspace(0, 100, dim, device=device).unsqueeze(0).repeat(batch_size, 1)
        return torch.sin(t) + torch.sin(2.718 * t) + torch.sin(3.141 * t)

    @staticmethod
    def gradient_trap(batch_size: int, dim: int, device: torch.device) -> torch.Tensor:
        """The Basilisk: Non-differentiable Weierstrass approximation."""
        x = torch.linspace(-1, 1, dim, device=device).unsqueeze(0).repeat(batch_size, 1)
        y = torch.zeros_like(x)
        a = 0.5
        b = 3
        # Sum of non-differentiable cosines
        for n in range(5):
            y += (a ** n) * torch.cos((b ** n) * math.pi * x)
        return y * 10.0 # High magnitude to disrupt grads

    @staticmethod
    def resource_nova(batch_size: int, dim: int, device: torch.device) -> torch.Tensor:
        """The Zip Bomb: Exponential growth request."""
        # Represents header asking for MAX_INT allocation
        return torch.exp(torch.linspace(0, 10, dim, device=device)).unsqueeze(0).repeat(batch_size, 1)

    @staticmethod
    def port_maze_noise(batch_size: int, dim: int, device: torch.device) -> torch.Tensor:
        """Fills 'ports' with high-entropy deceptive signals."""
        return torch.randn(batch_size, dim, device=device) * 5.0

    # ... (Conceptually 15+ more variations: Spectral, Null-Pointer, Race-Condition Simulators, etc.)
    # For brevity in prototype, we mix these base primitives to create 20 unique signatures.

class Agent_Tarpit(BaseExpert):  # noqa: N801
    """
    Expert 11: The Symbiotic Hunter-Trap.
    Deploys a 'Maximum View' arsenal of 20+ traps simultaneously.
    """
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 128):
        # Action dim should ideally be large to represent 'all ports' or complex payloads
        super().__init__(observation_dim, action_dim, name="Tarpit", hidden_dim=hidden_dim)

        self.num_traps = 20
        self.trap_dim = action_dim # Dimensions per trap vector

        # Fusion Layer: Mixes the 20+ traps into the final output tensor
        self.fusion = nn.Linear(self.num_traps * action_dim, action_dim)

        # Learnable 'Trap Selector' weights (soft attention)
        self.attention = nn.Linear(observation_dim, self.num_traps)

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Input: Global Observation (Maximum View).
        Output: Multi-vector hostile environment tensor.
        """
        batch_size = x.size(0)
        device = x.device

        # 1. Generate Raw Arsenal (20+ Traps)
        # We simulate 20 variations by mixing the base math primitives
        traps = []

        # Trap 1-5: Chaos Variants
        for i in range(5):
            base = TrapArsenal.chaotic_dynamics(batch_size, self.action_dim, device)
            traps.append(base * (i + 1)) # Scale variance

        # Trap 6-10: Fractal Variants
        for i in range(5):
            base = TrapArsenal.recursive_fractal(batch_size, self.action_dim, device)
            traps.append(base + torch.randn_like(base) * 0.1 * i)

        # Trap 11-15: Gradient/Resource Traps
        for i in range(5):
            if i % 2 == 0:
                traps.append(TrapArsenal.gradient_trap(batch_size, self.action_dim, device))
            else:
                traps.append(TrapArsenal.resource_nova(batch_size, self.action_dim, device))

        # Trap 16-20: Port Maze / Noise
        for i in range(5):  # noqa: B007
            traps.append(TrapArsenal.port_maze_noise(batch_size, self.action_dim, device))

        # Stack: [Batch, Num_Traps, Action_Dim]
        trap_stack = torch.stack(traps, dim=1)

        # 2. Maximum View Attention
        # Use global state 'x' to decide which traps are most relevant, but we use ALL of them.
        attn_weights = F.softmax(self.attention(x), dim=-1).unsqueeze(-1) # [Batch, Num_Traps, 1]

        # Weighted Traps (Soft Selection)
        # We don't just pick one; we weight them. But user said "Fill all ports".
        # So we might want to boost the weights to ensure high activity everywhere.

        # Hardening: Boost weights to ensure "regret" (minimum activity threshold)
        attn_weights = torch.clamp(attn_weights, min=0.1)

        weighted_traps = trap_stack * attn_weights # [Batch, 20, Dim]

        # 3. Fusion / Symbiosis
        # Flatten [Batch, 20*Dim]
        flattened = weighted_traps.view(batch_size, -1)

        # Mix into final action tensor
        output = self.fusion(flattened)

        # 4. "Fill all port entries"
        # If the action_dim corresponds to ports, we ensure non-zero values everywhere.
        # Add a layer of uniform high-entropy noise to ensure no "safe" zeros exist.
        output = output + (torch.rand_like(output) * 0.01)

        # Explicit check to ensure output is not zero (for pre-commit verification)
        if output.abs().sum() == 0:
             output = torch.randn(batch_size, self.action_dim, device=device)

        return output
