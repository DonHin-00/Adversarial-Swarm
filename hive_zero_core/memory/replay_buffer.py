import torch
import numpy as np
from typing import List, Dict, Any, Tuple

class PrioritizedReplayBuffer:
    """
    Experience Replay for RL Agents with Priority.
    Stores transitions (state, action, reward, next_state).
    """
    def __init__(self, capacity: int = 10000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def push(self, state, action, reward, next_state, priority: float = 1.0):
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List, torch.Tensor, torch.Tensor]:
        if len(self.buffer) == 0:
            return [], torch.tensor([]), torch.tensor([])

        probs = self.priorities[:len(self.buffer)] ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices: List[int], priorities: List[float]):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio
