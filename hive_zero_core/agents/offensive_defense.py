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

        # Signal Amplification Network
        # Non-linear amplification to ensure the reflected signal is stronger than the input
        self.amplifier = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(), # Preserve sign but saturate
            nn.Linear(hidden_dim, action_dim)
        )

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Reflects the attack vector with amplification.
        """
        # 1. Capture the signal (x)
        # 2. Amplify
        amplified = self.amplifier(x)

        # 3. Add Kinetic Energy (Cubic expansion for high-value spikes)
        # If input was high, output becomes massive.
        # x + x^3 behavior
        kinetic = amplified + torch.pow(amplified, 3)

        # 4. Invert Phase (Counter-Strike)
        # We don't just echo; we invert the signal to cancel/disrupt the attacker's wave
        reflection = kinetic * -10.0

        return reflection

class Agent_Flashbang(BaseExpert):
    """
    Expert 13: The Overloader.
    'Fast Showing': Generates hyper-velocity data bursts to crash ingestion.
    """
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__(observation_dim, action_dim, name="Flashbang", hidden_dim=hidden_dim)

        # Burst Generator (Dilated Convolution simulation via Linear expansion)
        # We want to take a small seed and explode it into a massive tensor
        self.expander = nn.Linear(observation_dim, hidden_dim * 4)
        self.formatter = nn.Linear(hidden_dim * 4, action_dim)

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generates sensory overload.
        """
        batch_size = x.size(0)

        # 1. Expand: Explosion phase
        expanded = F.relu(self.expander(x))

        # 2. Dazzle: Add random high-variance noise (The "Flash")
        flash = torch.randn_like(expanded) * 100.0

        # 3. Format: Compress back to output dim but keep the energy
        # The result is a dense, high-magnitude tensor that looks like valid but extreme signals
        out = self.formatter(expanded + flash)

        # Hardening: Ensure it's never zero (always blinding)
        out = out + 1.0

        return out

class Agent_Wraith(BaseExpert):
    """
    Expert 14: The Haunting.
    Persistence / Hack Back.
    Embeds self-replicating polymorphic shellcode patterns into the noise.
    """
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__(observation_dim, action_dim, name="Wraith", hidden_dim=hidden_dim)

        # Polymorphic Engine (GAN-like)
        # Generates "Payload" tensors that look different every time but contain the "Beacon" signature
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid() # Normalize to [0,1] bits
        )

        self.decoder = nn.Linear(hidden_dim, action_dim)

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generates the persistence payload.
        """
        batch_size = x.size(0)

        # 1. Generate the "Beacon" seed from current state
        # (Adapts to environment so it looks like local data)
        seed = self.encoder(x)

        # 2. Add Polymorphic Salt (Random Noise)
        # Ensures signature changes every execution
        salt = torch.rand_like(seed)
        payload_code = seed * salt

        # 3. Decode into Action Space
        # Represents the obfuscated shellcode byte-stream
        beacon = self.decoder(payload_code)

        # 4. Stealth Coating
        # Add a layer of "normalcy" (small values) to hide the sharp "exploit" spikes
        # The exploit is hidden in the variance, not the mean
        stealth_beacon = beacon * 5.0 # Boost signal strength for transmission

        return stealth_beacon
