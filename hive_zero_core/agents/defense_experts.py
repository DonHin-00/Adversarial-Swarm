import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hive_zero_core.agents.base_expert import BaseExpert


class TrapArsenal:
    """
    Expert-Level Adversarial Mathematics Library.

    Generates 25+ distinct high-fidelity trap signatures from six
    primitive families: chaotic, fractal, gradient-trap, resource-nova,
    port-maze noise, spectral, and temporal.
    """

    @staticmethod
    def chaotic_dynamics(batch_size: int, dim: int, device: torch.device) -> torch.Tensor:
        """The Siren: Logistic Map Chaos Generator."""
        r = 3.99  # Chaotic regime
        x = torch.rand(batch_size, dim, device=device)
        for _ in range(5):
            x = r * x * (1 - x)
        return x

    @staticmethod
    def recursive_fractal(batch_size: int, dim: int, device: torch.device) -> torch.Tensor:
        """The Infinite Hallway: Self-similar expansion via high-freq sinusoidal interference."""
        t = torch.linspace(0, 100, dim, device=device).unsqueeze(0).expand(batch_size, -1)
        return torch.sin(t) + torch.sin(2.718 * t) + torch.sin(3.141 * t)

    @staticmethod
    def gradient_trap(batch_size: int, dim: int, device: torch.device) -> torch.Tensor:
        """The Basilisk: Non-differentiable Weierstrass approximation."""
        x = torch.linspace(-1, 1, dim, device=device).unsqueeze(0).expand(batch_size, -1)
        y = torch.zeros_like(x)
        a, b = 0.5, 3
        for n in range(5):
            y += (a ** n) * torch.cos((b ** n) * math.pi * x)
        return y * 10.0

    @staticmethod
    def resource_nova(batch_size: int, dim: int, device: torch.device) -> torch.Tensor:
        """The Zip Bomb: Exponential growth request."""
        return torch.exp(torch.linspace(0, 10, dim, device=device)).unsqueeze(0).expand(batch_size, -1)

    @staticmethod
    def port_maze_noise(batch_size: int, dim: int, device: torch.device) -> torch.Tensor:
        """High-entropy deceptive port signals."""
        return torch.randn(batch_size, dim, device=device) * 5.0

    @staticmethod
    def spectral_comb(batch_size: int, dim: int, device: torch.device) -> torch.Tensor:
        """Spectral Comb: Periodic spikes at harmonic frequencies to confuse FFT-based detectors."""
        t = torch.linspace(0, 2 * math.pi, dim, device=device).unsqueeze(0).expand(batch_size, -1)
        comb = torch.zeros_like(t)
        for k in range(1, 8):
            comb += torch.cos(k * 7 * t) / k
        return comb * 3.0

    @staticmethod
    def temporal_jitter(batch_size: int, dim: int, device: torch.device) -> torch.Tensor:
        """Temporal Jitter: Exponentially-distributed inter-arrival noise to defeat timing analysis."""
        # Exponential distribution via inverse CDF: -ln(U) / λ
        u = torch.clamp(torch.rand(batch_size, dim, device=device), min=1e-7)
        return -torch.log(u) * 2.0



class Agent_Tarpit(BaseExpert):
    """
    Expert 11: Multi-Head Attention Symbiotic Hunter-Trap

    Deploys a 'Maximum View' arsenal of 25 traps across seven primitive
    families. Uses multi-head cross-attention (observation queries, trap
    keys/values) for adaptive fusion instead of a flat linear layer,
    producing richer context-dependent trap combinations.
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 128,
                 num_traps: int = 25, attn_heads: int = 4):
        super().__init__(observation_dim, action_dim, name="Tarpit", hidden_dim=hidden_dim)

        self.num_traps = num_traps

        # Project traps to a common hidden space for attention
        self.trap_proj = nn.Linear(action_dim, hidden_dim)
        self.obs_proj = nn.Linear(observation_dim, hidden_dim)

        # Multi-head cross-attention: observation attends over trap keys/values
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=attn_heads, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def _generate_arsenal(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate all trap vectors and stack as [B, num_traps, action_dim]."""
        traps = []
        dim = self.action_dim

        # Family 1: Chaos (5 variants)
        for i in range(5):
            traps.append(TrapArsenal.chaotic_dynamics(batch_size, dim, device) * (i + 1))

        # Family 2: Fractal (4 variants)
        for i in range(4):
            base = TrapArsenal.recursive_fractal(batch_size, dim, device)
            traps.append(base + torch.randn_like(base) * 0.1 * i)

        # Family 3: Gradient / Resource (4 variants)
        for i in range(4):
            if i % 2 == 0:
                traps.append(TrapArsenal.gradient_trap(batch_size, dim, device))
            else:
                traps.append(TrapArsenal.resource_nova(batch_size, dim, device))

        # Family 4: Port Maze (4 variants)
        for _ in range(4):
            traps.append(TrapArsenal.port_maze_noise(batch_size, dim, device))

        # Family 5: Spectral (4 variants)
        for i in range(4):
            traps.append(TrapArsenal.spectral_comb(batch_size, dim, device) * (0.5 + 0.5 * i))

        # Family 6: Temporal (4 variants)
        for i in range(4):
            traps.append(TrapArsenal.temporal_jitter(batch_size, dim, device) * (1 + i))

        assert len(traps) >= self.num_traps, (
            f"Generated {len(traps)} traps but need {self.num_traps}"
        )
        return torch.stack(traps[:self.num_traps], dim=1)  # [B, T, D]

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor],
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.size(0)
        device = x.device

        # Generate arsenal
        trap_stack = self._generate_arsenal(batch_size, device)  # [B, T, D]

        # Project to hidden for attention
        trap_kv = self.trap_proj(trap_stack)   # [B, T, H]
        obs_q = self.obs_proj(x).unsqueeze(1)  # [B, 1, H]

        # Cross-attention: observation queries attend over traps
        attn_out, _ = self.cross_attn(obs_q, trap_kv, trap_kv)  # [B, 1, H]
        attn_out = self.attn_norm(attn_out + obs_q)  # Residual
        attn_out = attn_out.squeeze(1)  # [B, H]

        # Output projection
        output = self.output_proj(attn_out)

        # Ensure non-zero: add small uniform noise so no port is "safe"
        output = output + torch.rand_like(output) * 0.01

        # Numerical safety: if output is effectively zero, inject noise
        if output.abs().sum() < 1e-6:
            output = output + torch.randn_like(output) * 0.1

        return output
