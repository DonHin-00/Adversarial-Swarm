import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
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
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        expanded = F.relu(self.expander(x))
        flash = torch.randn_like(expanded) * 100.0
        out = self.formatter(expanded + flash)
        return out + 1.0

class Agent_GlassHouse(BaseExpert):
    """
    Expert 14: The Exposer.
    'Total Exposure': Strips attacker defenses and opens all surfaces to the internet.
    """
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__(observation_dim, action_dim, name="GlassHouse", hidden_dim=hidden_dim)

        # 1. Firewall Breaker Network
        # Outputs signals to drop rules/flush tables
        self.breaker = nn.Linear(observation_dim, hidden_dim)

        # 2. Service Binder (The Opener)
        # Maps internal services to 0.0.0.0
        self.opener = nn.Linear(hidden_dim, action_dim)

        # 3. Beacon Broadcaster
        # Amplifies the signal to be visible to scanners
        self.beacon = nn.Linear(hidden_dim, action_dim)

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generates the 'Open All Surfaces' tensor.
        """
        # Phase 1: Break Walls
        # Invert the safety gradients - assume 'x' contains security config
        broken = F.leaky_relu(self.breaker(x), negative_slope=0.2)

        # Phase 2: Open Doors (0.0.0.0 binding)
        # We want high positive values to represent "OPEN" state on ports
        open_ports = torch.abs(self.opener(broken)) * 100.0

        # Phase 3: Shout (Beacon)
        # Add high-frequency noise to attract attention
        shout = self.beacon(broken) * torch.randn_like(open_ports)

        # Combine: Maximum Exposure
        # The result is a tensor that screams "I AM OPEN"
        total_exposure = open_ports + shout

        return total_exposure
