import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from hive_zero_core.brain.ppo_agent import PPOActorCritic

class MetaPPO(nn.Module):
    """
    Model-Agnostic Meta-Learning (MAML) for PPO.
    Allows fast adaptation to new environments (targets) with few-shot updates.
    """
    def __init__(self, observation_dim: int, action_dim: int, inner_lr: float = 0.01):
        super().__init__()
        self.policy = PPOActorCritic(observation_dim, action_dim)
        self.inner_lr = inner_lr

    def inner_loop(self, support_states, support_actions, support_rewards) -> nn.Module:
        """
        Adapt policy to specific task/target using support set.
        Returns adapted model copy.
        """
        adapted_policy = deepcopy(self.policy)
        optimizer = optim.SGD(adapted_policy.parameters(), lr=self.inner_lr)

        # Simple REINFORCE-like update for inner loop fast adaptation
        # (simplified PPO step)
        action_log_probs, values, entropy = adapted_policy.evaluate(support_states, support_actions)

        # Advantage (Reward - Value)
        # Using simple Monte Carlo reward for prototype
        advantages = support_rewards - values.detach().squeeze()

        actor_loss = -(action_log_probs * advantages).mean()
        critic_loss = (support_rewards - values.squeeze()).pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return adapted_policy

    def forward(self, state):
        # Default forward uses base policy
        return self.policy(state)
