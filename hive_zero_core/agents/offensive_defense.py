from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hive_zero_core.agents.base_expert import BaseExpert


class Agent_FeedbackLoop(BaseExpert):
    """
    Expert 12: The Reflector.
    Captures attacker signals, amplifies them, and re-injects them.
    "Stop hitting yourself."
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__(observation_dim, action_dim, name="FeedbackLoop", hidden_dim=hidden_dim)
        self.amplifier = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, action_dim)
        )

    def _forward_impl(
        self, x: torch.Tensor, context: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        amplified = self.amplifier(x)
        kinetic = amplified + torch.pow(amplified, 3)
        return kinetic * -10.0


class Agent_Flashbang(BaseExpert):
    """
    Expert 13: The Overloader.
    'Fast Showing': Generates hyper-velocity data bursts to crash ingestion.
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__(observation_dim, action_dim, name="Flashbang", hidden_dim=hidden_dim)
        self.expander = nn.Linear(observation_dim, hidden_dim * 4)
        self.formatter = nn.Linear(hidden_dim * 4, action_dim)

    def _forward_impl(
        self, x: torch.Tensor, context: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        expanded = F.relu(self.expander(x))
        flash = torch.randn_like(expanded) * 100.0
        out = self.formatter(expanded + flash)
        return out + 1.0


class Agent_GlassHouse(BaseExpert):
    """
    Expert 14: The Exposer (Holographic Edition).
    'Total Exposure': Strips attacker defenses and opens all surfaces to the internet.
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__(observation_dim, action_dim, name="GlassHouse", hidden_dim=hidden_dim)

        # 1. Firewall Breaker
        self.breaker = nn.Linear(observation_dim, hidden_dim)

        # 2. Holographic Service Binder
        # Generates complex-valued signals (Real + Imaginary) to simulate Quantum Port States
        # Real part = Physical Binding (0.0.0.0)
        # Imaginary part = Decoy State (Honeyport)
        self.opener_real = nn.Linear(hidden_dim, action_dim)
        self.opener_imag = nn.Linear(hidden_dim, action_dim)

        # 3. Beacon Broadcaster
        self.beacon = nn.Linear(hidden_dim, action_dim)

    def _forward_impl(
        self, x: torch.Tensor, context: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generates the 'Holographic Total Exposure' tensor.
        """
        # Phase 1: Break Walls
        broken = F.leaky_relu(self.breaker(x), negative_slope=0.2)

        # Phase 2: Quantum Port Binding (Holographic)
        real_ports = torch.abs(self.opener_real(broken)) * 100.0  # High value = OPEN
        imag_ports = torch.sin(self.opener_imag(broken)) * 50.0  # Phase shift = TRAP

        # Phase 3: Shout (Beacon)
        shout = self.beacon(broken) * torch.randn_like(real_ports)

        # Fusion: Magnitude of the complex signal
        # |Z| = sqrt(Real^2 + Imag^2)
        # This creates a non-linear, high-energy exposure surface that is mathematically consistent
        # but operationally devastating.
        total_exposure = torch.sqrt(torch.pow(real_ports, 2) + torch.pow(imag_ports, 2) + 1e-6)

        # Add the Beacon noise
        final_output = total_exposure + shout

        return final_output
