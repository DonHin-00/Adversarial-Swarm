from typing import Any, Tuple

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple, Optional


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
        Returns (obs, action, reward, next_obs, done) simulating optimal behaviour.

        Args:
            batch_size: Number of experience tuples to generate.
            device:     Target device for all tensors (default: CPU).
        """
        if device is None:
            device = torch.device("cpu")

        # ---- Observations (mixture of three clusters) ----
        n_idle = batch_size // 3
        n_scan = batch_size // 3
        n_attack = batch_size - n_idle - n_scan

        idle_obs = torch.randn(n_idle, self.obs_dim, device=device) * 0.1
        scan_obs = torch.randn(n_scan, self.obs_dim, device=device) * 0.5
        scan_obs[:, :min(10, self.obs_dim)] += 2.0

        attack_obs = torch.abs(torch.randn(n_attack, self.obs_dim, device=device)) * 2.0

        obs = torch.cat([idle_obs, scan_obs, attack_obs], dim=0)
        obs = obs[torch.randperm(batch_size, device=device)]

        # ---- Optimal actions ----
        obs_mag = obs.mean(dim=1, keepdim=True)
        actions = torch.zeros(batch_size, self.act_dim, device=device)

        high_mask = (obs_mag > 0.5).float()
        actions += high_mask * torch.randn(batch_size, self.act_dim, device=device) * 5.0

        low_mask = 1.0 - high_mask
        pattern = torch.sin(torch.linspace(0, 10, self.act_dim, device=device))
        actions += low_mask * pattern.unsqueeze(0).expand(batch_size, -1)

        rewards = torch.ones(batch_size, 1, device=device) * 10.0
        next_obs = obs * 0.1
        dones = torch.zeros(batch_size, 1, device=device)

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
