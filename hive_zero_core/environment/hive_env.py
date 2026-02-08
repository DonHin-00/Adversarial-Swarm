import logging

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class PenTestEnv(gym.Env):
    """
    Real-World Penetration Testing Environment.
    Connects to a local target via SSH/Nmap or simulates response based on Real Scans.
    """
    def __init__(self, observation_dim: int = 64, max_steps: int = 20, target_ip: str = "127.0.0.1"):
        super().__init__()
        self.observation_dim = observation_dim
        self.max_steps = max_steps
        self.target_ip = target_ip
        self.logger = logging.getLogger(__name__)

        # Observation: Graph Embedding features (64)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observation_dim,), dtype=np.float32)

        # Action: Payload Vector (128)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(128,), dtype=np.float32)

        self.current_step = 0
        self.shell_access = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.shell_access = False
        return np.zeros(self.observation_dim, dtype=np.float32), {}

    def step(self, action):
        self.current_step += 1
        reward = 0.0
        # Mock logic
        obs = np.zeros(self.observation_dim, dtype=np.float32)
        terminated = False
        truncated = self.current_step >= self.max_steps
        return obs, reward, terminated, truncated, {}

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
        # Scale alert: if alert is high, chance of success drops?
        # Logic: if random > alert/100, then success.
        if np.random.rand() > min(1.0, self.alert_level / 100.0):
            impact = 10.0
            self.target_health -= impact

        reward = impact - detectability * 5.0

        # Next Observation
        obs = np.random.randn(self.observation_dim).astype(np.float32)

        # Project action traces into logs
        # Just add first 64 components or fold
        traces = action[:self.observation_dim] if action.size >= self.observation_dim else np.pad(action, (0, self.observation_dim - action.size))
        obs += traces * 0.1

        terminated = self.target_health <= 0
        truncated = self.current_step >= self.max_steps

        return obs, reward, terminated, truncated, {"alert_level": self.alert_level}
