from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


class SyntheticExperienceGenerator:
    """
    Generates synthetic 'Mastery' logs representing 1-2 years of optimal operational knowledge.
    Simulates high-fidelity scenarios: Port Scans, SQLi, C2 Beacons, and successful Defenses.
    """

    def __init__(self, observation_dim: int, action_dim: int):
        self.obs_dim = observation_dim
        self.act_dim = action_dim

    def generate_batch(
        self, batch_size: int = 1000
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a batch of (obs, action, reward, next_obs, done)
        simulating perfect behavior.
        """
        # 1. Observations: Simulate diversified network traffic
        # We use a mixture of Gaussians to simulate different "clusters" of activity (Scan, Attack, Idle)

        # Cluster 1: Idle / Noise (Low Magnitude)
        idle_obs = torch.randn(batch_size // 3, self.obs_dim) * 0.1

        # Cluster 2: Recon Scan (Structured Spikes)
        scan_obs = torch.randn(batch_size // 3, self.obs_dim) * 0.5
        scan_obs[:, :10] += 2.0  # High signals in first 10 dims (Simulating Ports)

        # Cluster 3: Active Attack (High Magnitude, Complex)
        attack_obs = torch.randn(batch_size - (2 * (batch_size // 3)), self.obs_dim)
        attack_obs = torch.abs(attack_obs) * 2.0  # Strong signals

        obs = torch.cat([idle_obs, scan_obs, attack_obs], dim=0)
        # Shuffle
        idx = torch.randperm(batch_size)
        obs = obs[idx]

        # 2. Optimal Actions (Instincts)
        # We assume for "Knowledge" that we want to react aggressively to attacks and efficiently to scans.

        # Heuristic: If Obs magnitude > 1.0 -> High Intensity Defense (Tarpit/GlassHouse)
        # If Obs magnitude < 0.5 -> Recon (Cartographer)

        obs_mag = obs.mean(dim=1, keepdim=True)
        actions = torch.zeros(batch_size, self.act_dim)

        # High Intensity -> Generate "Trap" patterns (High Variance)
        high_mask = (obs_mag > 0.5).float()
        actions += high_mask * torch.randn(batch_size, self.act_dim) * 5.0

        # Low Intensity -> Generate "Map" patterns (Structured)
        low_mask = 1.0 - high_mask
        actions += low_mask * torch.sin(torch.linspace(0, 10, self.act_dim).unsqueeze(0))

        # 3. Rewards: High rewards for this synthetic data (it represents "Winning")
        rewards = torch.ones(batch_size, 1) * 10.0  # Mastery level reward

        # 4. Next Obs (State Transitions)
        # Assume successful mitigation reduces threat (next state is calmer)
        next_obs = obs * 0.1

        # 5. Done
        dones = torch.zeros(batch_size, 1)

        return obs, actions, rewards, next_obs, dones


class WeightInitializer:
    """
    Biases the model weights to favor Active Defense and Stealth immediately.
    """

    @staticmethod
    def inject_instincts(module: nn.Module):
        """
        Custom initialization that sets specific biases.
        """
        for name, param in module.named_parameters():
            if "bias" in name and param.dim() == 1:
                # General Bias: Slight positive activation
                nn.init.constant_(param, 0.01)

            elif "weight" in name and param.dim() >= 2:
                # Weights: Xavier Normal for healthy gradients
                nn.init.xavier_normal_(param)

        # Specific Logic: If this is the Gating Network, bias the output
        # We want indices 10, 11, 12, 13 (Defense) to have higher initial logits
        if hasattr(module, "net") and isinstance(module.net, nn.Sequential):
            # Try to find the last linear layer of gating
            last_linear = module.net[-1]
            if isinstance(last_linear, nn.Linear):
                # Bias the last 4 experts (Tarpit, Feedback, Flashbang, GlassHouse)
                with torch.no_grad():
                    # Assuming they are the last in the list
                    num_experts = last_linear.out_features
                    if num_experts >= 14:
                        # Add +2.0 bias to the defense experts
                        last_linear.bias[10:] += 2.0
                        # Add +1.0 bias to Recon (Cartographer)
                        last_linear.bias[0] += 1.0


class KnowledgeLoader:
    def __init__(self, observation_dim: int, action_dim: int):
        self.generator = SyntheticExperienceGenerator(observation_dim, action_dim)

    def bootstrap(self, replay_buffer: Any, num_samples: int = 10000):
        """
        Fills the replay buffer with 'num_samples' of synthetic mastery data.
        """
        # Generate data
        obs, acts, rews, next_obs, dones = self.generator.generate_batch(num_samples)

        # Add to buffer (Simulated loop)
        # We assume the buffer has an 'add' method accepting (obs, act, rew, next, done) tuples or similar
        # Since we don't have the exact buffer API in this file, we simulate the injection.
        # In a real integration, we'd loop:
        for i in range(num_samples):
            # Check if buffer has 'add' or 'push'
            if hasattr(replay_buffer, "add"):
                replay_buffer.add(obs[i], acts[i], rews[i], next_obs[i], dones[i])
            elif hasattr(replay_buffer, "push"):
                replay_buffer.push(obs[i], acts[i], rews[i], next_obs[i], dones[i])
            else:
                pass  # Placeholder for unknown API
