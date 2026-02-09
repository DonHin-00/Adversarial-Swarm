import torch
import numpy as np
from typing import List, Any, Tuple, Union

class PrioritizedReplayBuffer:
    """
    Experience Replay Buffer for HIVE-ZERO Agents with Proportional Prioritization.

    Stores and samples transitions based on their temporal difference (TD) error
    to improve training efficiency in sparse reward environments.
    """
    def __init__(self, capacity: int = 10000, alpha: float = 0.6):
        """
        Args:
            capacity: Maximum number of transitions to store.
            alpha: Randomization constant (0 = uniform, 1 = full prioritization).
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def push(self, state: Any, action: Any, reward: float, next_state: Any):
        """
        Adds a transition to the buffer.
        """
        # Set max priority for new transitions to ensure they are sampled at least once
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Tuple], np.ndarray, torch.Tensor]:
        """
        Samples a batch of transitions.

        Args:
            batch_size: Number of transitions to sample.
            beta: Importance-sampling constant (0 = no correction, 1 = full correction).

        Returns:
            samples: List of transitions.
            indices: Indices of sampled transitions.
            weights: Importance-sampling weights for the loss function.
        """
        if len(self.buffer) == 0:
            return [], np.array([]), torch.tensor([])

        # Handle smaller-than-requested batches
        current_size = len(self.buffer)
        batch_size = min(batch_size, current_size)

        # Proportional sampling: P(i) = p_i^alpha / sum(p_k^alpha)
        probs = self.priorities[:current_size] ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(current_size, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # Importance Sampling (IS) weights to correct for bias
        # w_i = (1/N * 1/P(i))^beta
        weights = (current_size * probs[indices]) ** (-beta)
        weights /= weights.max() # Normalize weights

        return samples, indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices: Union[List[int], np.ndarray], priorities: Union[List[float], np.ndarray]):
        """
        Updates priorities for sampled transitions after a training step.
        """
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio + 1e-6 # Add epsilon to avoid zero priority

    def __len__(self) -> int:
        return len(self.buffer)
