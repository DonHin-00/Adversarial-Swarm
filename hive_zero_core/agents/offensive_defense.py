from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hive_zero_core.agents.base_expert import BaseExpert


class Agent_FeedbackLoop(BaseExpert):
    """
    Expert 12: Multi-Scale Spectral Reflector

    Captures attacker signals across multiple frequency bands, amplifies
    each independently via a learned gain schedule, and re-injects the
    composite at high energy.  The multi-scale decomposition ensures both
    low-frequency command-and-control traffic and high-frequency probes
    are reflected back effectively.
    """

    def __init__(
        self, observation_dim: int, action_dim: int, hidden_dim: int = 128, num_scales: int = 3
    ):
        super().__init__(observation_dim, action_dim, name="FeedbackLoop", hidden_dim=hidden_dim)
        self.num_scales = num_scales

        # Per-scale amplifier branches
        self.scale_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(observation_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, action_dim),
                )
                for _ in range(num_scales)
            ]
        )

        # Learnable per-scale gain (initialised to increasing magnitude)
        self.gains = nn.Parameter(torch.linspace(1.0, 10.0, num_scales))

        # Fusion head
        self.fusion = nn.Linear(action_dim * num_scales, action_dim)
        self.norm = nn.LayerNorm(action_dim)

    def _forward_impl(
        self, x: torch.Tensor, context: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        branches = []
        for i, net in enumerate(self.scale_nets):
            amp = net(x)
            # Non-linear amplification: signal + cubic harmonic at learned gain
            kinetic = amp + torch.pow(amp, 3)
            branches.append(kinetic * -self.gains[i])

        fused = self.fusion(torch.cat(branches, dim=-1))
        return self.norm(fused)


class Agent_Flashbang(BaseExpert):
    """
    Expert 13: Adaptive Burst-Schedule Overloader

    Generates hyper-velocity data bursts calibrated to crash target ingestion
    pipelines.  A learned burst-schedule modulates the noise magnitude
    dynamically based on the observation, replacing the fixed 100× noise
    multiplier with context-sensitive intensity.
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__(observation_dim, action_dim, name="Flashbang", hidden_dim=hidden_dim)

        self.expander = nn.Linear(observation_dim, hidden_dim * 4)
        self.norm = nn.LayerNorm(hidden_dim * 4)
        self.formatter = nn.Linear(hidden_dim * 4, action_dim)

        # Adaptive burst intensity: learns how much noise to inject
        self.intensity_head = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # Ensures positive intensity
        )

    def _forward_impl(
        self, x: torch.Tensor, context: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        expanded = F.relu(self.norm(self.expander(x)))

        # Context-adaptive noise intensity instead of fixed 100×
        intensity = self.intensity_head(x) + 1.0  # minimum intensity of 1.0
        flash = torch.randn_like(expanded) * intensity

        out = self.formatter(expanded + flash)

        # Bias offset: ensures output is never zero, maintaining persistent
        # overload pressure even when the formatter produces near-zero values
        return out + 1.0


class Agent_GlassHouse(BaseExpert):
    """
    Expert 14: Learnable Phase-Modulated Exposer

    Strips attacker defences by generating a holographic exposure surface
    with learnable phase modulation.  The real/imaginary port signals now
    use multi-layer networks with learned phase offsets, producing richer
    interference patterns that are harder to filter.
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__(observation_dim, action_dim, name="GlassHouse", hidden_dim=hidden_dim)

        # Firewall breaker with layer norm
        self.breaker = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Complex-valued port binders with learned phase
        self.opener_real = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.opener_imag = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Learnable phase offset per port dimension
        self.phase_offset = nn.Parameter(torch.zeros(1, action_dim))

        # Beacon broadcaster with adaptive scaling
        self.beacon = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.beacon_scale = nn.Parameter(torch.ones(1))

    def _forward_impl(
        self, x: torch.Tensor, context: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Phase 1: Break Walls
        broken = F.leaky_relu(self.breaker(x), negative_slope=0.2)

        # Phase 2: Holographic port binding with learned phase modulation
        real_ports = torch.abs(self.opener_real(broken)) * 100.0
        imag_raw = self.opener_imag(broken)
        imag_ports = torch.sin(imag_raw + self.phase_offset) * 50.0

        # Phase 3: Beacon with learned scale
        shout = self.beacon(broken) * self.beacon_scale * torch.randn_like(real_ports)

        # Fusion: |Z| = sqrt(Re² + Im²)
        total_exposure = torch.sqrt(torch.pow(real_ports, 2) + torch.pow(imag_ports, 2) + 1e-6)

        return total_exposure + shout
