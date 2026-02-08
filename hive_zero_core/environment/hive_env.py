import gymnasium as gym
from gymnasium import spaces
import torch
import numpy as np
from typing import Dict, Any, List
from hive_zero_core.hive_mind import HiveMind

class HiveZeroEnv(gym.Env):
    """
    Gymnasium Wrapper for HIVE-ZERO.
    """
    def __init__(self, observation_dim: int = 64, max_steps: int = 100):
        super().__init__()
        self.observation_dim = observation_dim
        self.max_steps = max_steps
        self.current_step = 0

        # Obs: Log Stream
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observation_dim,), dtype=np.float32)

        # Action: Packet Vector (128)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(128,), dtype=np.float32)

        # Internal State
        self.target_health = 100.0
        self.alert_level = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.target_health = 100.0
        self.alert_level = 0.0

        # Initial observation (Normal logs)
        obs = np.random.randn(self.observation_dim).astype(np.float32)
        return obs, {}

    def step(self, action):
        self.current_step += 1

        # Simulation Logic
        detectability = np.linalg.norm(action) * 0.1
        self.alert_level += detectability

        impact = 0.0
        if np.random.rand() > min(1.0, self.alert_level / 100.0): # Scale alert
            impact = 10.0
            self.target_health -= impact

        reward = impact - detectability * 5.0

        # 4. Next Observation
        obs = np.random.randn(self.observation_dim).astype(np.float32)

        # Project action traces into logs
        # Action (128) -> Obs (64)
        # Just add first 64 components or fold
        traces = action[:self.observation_dim] if action.size >= self.observation_dim else np.pad(action, (0, self.observation_dim - action.size))
        obs += traces * 0.1

        terminated = self.target_health <= 0
        truncated = self.current_step >= self.max_steps

        return obs, reward, terminated, truncated, {"alert_level": self.alert_level}
