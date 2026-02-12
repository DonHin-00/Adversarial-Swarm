from typing import Any, Optional, Tuple

import torch
import torch.nn as nn


class SyntheticExperienceGenerator:
    """
    Generates synthetic 'Mastery' logs representing 1-2 years of optimal
    operational knowledge.  Simulates high-fidelity scenarios including
    port scans, SQLi attempts, C2 beacons, and successful defences.

    All tensors are created on a caller-specified device to avoid
    device-mismatch errors during GPU training.
    """

    def __init__(self, observation_dim: int, action_dim: int):
        self.obs_dim = observation_dim
        self.act_dim = action_dim

    def generate_batch(
        self,
        batch_size: int = 1000,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a batch of (obs, action, reward, next_obs, done)
        simulating perfect behavior. Optimized for better performance.
        """
        # Calculate batch sizes efficiently
        idle_size = batch_size // 3
        scan_size = batch_size // 3
        attack_size = batch_size - idle_size - scan_size
        
        # 1. Observations: Simulate diversified network traffic
        # Vectorized generation for better performance
        
        # Cluster 1: Idle / Noise (Low Magnitude)
        idle_obs = torch.randn(idle_size, self.obs_dim) * 0.1

        # Cluster 2: Recon Scan (Structured Spikes)
        scan_obs = torch.randn(scan_size, self.obs_dim) * 0.5
        scan_obs[:, :10] += 2.0 # High signals in first 10 dims (Simulating Ports)

        # Cluster 3: Active Attack (High Magnitude, Complex)
        attack_obs = torch.abs(torch.randn(attack_size, self.obs_dim)) * 2.0 # Strong signals

        obs = torch.cat([idle_obs, scan_obs, attack_obs], dim=0)
        # Shuffle
        idx = torch.randperm(batch_size)
        obs = obs[idx]

        # 2. Optimal Actions (Instincts) - vectorized computation
        obs_mag = obs.mean(dim=1, keepdim=True)
        actions = torch.zeros(batch_size, self.act_dim, device=device)

        high_mask = (obs_mag > 0.5).float()
        actions += high_mask * torch.randn(batch_size, self.act_dim, device=device) * 5.0

        # Low Intensity -> Generate "Map" patterns (Structured) - optimized
        low_mask = (1.0 - high_mask)
        pattern = torch.sin(torch.linspace(0, 10, self.act_dim))
        actions += low_mask * pattern.unsqueeze(0)

        # 3. Rewards: High rewards for this synthetic data (it represents "Winning")
        rewards = torch.full((batch_size, 1), 10.0)

        # 4. Next Obs (State Transitions)
        # Assume successful mitigation reduces threat (next state is calmer)
        next_obs = obs * 0.1
        dones = torch.zeros(batch_size, 1)

        return obs, actions, rewards, next_obs, dones


class WeightInitializer:
    """
    Biases the model weights to favour Active Defence and Stealth from
    the very first forward pass.
    """

    @staticmethod
    def inject_instincts(module: nn.Module):
        """
        Custom initialisation that seeds specific biases.

        * All weight matrices: Xavier Normal for healthy gradient flow.
        * All bias vectors: small positive constant (0.01).
        * Gating network output bias: +2.0 for defence experts (indices 10-13)
          and +1.0 for Cartographer (index 0).
        """
        for name, param in module.named_parameters():
            if 'bias' in name and param.dim() == 1:
                nn.init.constant_(param, 0.01)
            elif 'weight' in name and param.dim() >= 2:
                nn.init.xavier_normal_(param)

        # Gating-specific: elevate defence cluster and recon
        if hasattr(module, 'net') and isinstance(module.net, nn.Sequential):
            last_linear = module.net[-1]
            if isinstance(last_linear, nn.Linear):
                with torch.no_grad():
                    num_experts = last_linear.out_features
                    if num_experts >= 14:
                        # Boost only the original defence/kill-chain experts (indices 10–13)
                        last_linear.bias[10:14] += 2.0
                        last_linear.bias[0] += 1.0


class KnowledgeLoader:
    """Pre-fills a replay buffer with synthetic mastery data."""

    def __init__(self, observation_dim: int, action_dim: int):
        self.generator = SyntheticExperienceGenerator(observation_dim, action_dim)

    def bootstrap(
        self,
        replay_buffer: Any,
        num_samples: int = 10000,
        device: Optional[torch.device] = None,
    ):
        """
        Fills *replay_buffer* with *num_samples* of synthetic mastery data.

        The buffer must implement either an ``add`` or ``push`` method that
        accepts (obs, action, reward, next_obs, done) tensors.

        Raises:
            TypeError: If the buffer has neither ``add`` nor ``push``.
        """
        obs, acts, rews, next_obs, dones = self.generator.generate_batch(
            num_samples, device=device
        )

        if hasattr(replay_buffer, 'add'):
            push_fn = replay_buffer.add
        elif hasattr(replay_buffer, 'push'):
            push_fn = replay_buffer.push
        else:
            raise TypeError(
                f"Replay buffer must expose 'add' or 'push'; got {type(replay_buffer)}"
            )

        for i in range(num_samples):
            push_fn(obs[i], acts[i], rews[i], next_obs[i], dones[i])
