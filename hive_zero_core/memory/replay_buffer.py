import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int = 10000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer: List[Tuple[Any, Any, Any, Any]] = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def push(self, state, action, reward, next_state, priority: Optional[float] = None):
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state)

        # Use provided priority if given, otherwise use max_prio
        self.priorities[self.pos] = priority if priority is not None else max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Any], np.ndarray, torch.Tensor]:
        if len(self.buffer) == 0:
            return [], np.array([]), torch.tensor([])

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
