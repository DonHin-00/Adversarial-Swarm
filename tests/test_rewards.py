"""
Tests for reward calculation functions.
"""

import pytest
import torch

from hive_zero_core.training.rewards import CompositeReward


def test_composite_reward_initialization():
    """Test CompositeReward initialization with default weights."""
    reward_calc = CompositeReward()
    assert reward_calc.w_adv == 1.0
    assert reward_calc.w_info == 0.5
    assert reward_calc.w_stealth == 0.8


def test_composite_reward_custom_weights():
    """Test CompositeReward initialization with custom weights."""
    reward_calc = CompositeReward(w_adv=0.5, w_info=0.3, w_stealth=0.2)
    assert reward_calc.w_adv == 0.5
    assert reward_calc.w_info == 0.3
    assert reward_calc.w_stealth == 0.2


def test_adversarial_reward():
    """Test adversarial reward calculation."""
    reward_calc = CompositeReward()
    
    # High evasion probability should give high reward
    high_score = torch.tensor([0.9])
    reward = reward_calc.calculate_adversarial_reward(high_score)
    assert reward.item() == pytest.approx(0.9)
    
    # Low evasion probability should give low reward
    low_score = torch.tensor([0.1])
    reward = reward_calc.calculate_adversarial_reward(low_score)
    assert reward.item() == pytest.approx(0.1)


def test_info_gain_reward_positive():
    """Test information gain reward with positive gain."""
    reward_calc = CompositeReward()
    
    # Entropy decreased -> positive reward
    prev_entropy = 1.0
    current_entropy = 0.5
    reward = reward_calc.calculate_info_gain_reward(prev_entropy, current_entropy)
    assert reward == 0.5


def test_info_gain_reward_negative():
    """Test information gain reward with negative gain (no reward)."""
    reward_calc = CompositeReward()
    
    # Entropy increased -> no reward
    prev_entropy = 0.5
    current_entropy = 1.0
    reward = reward_calc.calculate_info_gain_reward(prev_entropy, current_entropy)
    assert reward == 0.0


def test_info_gain_reward_zero():
    """Test information gain reward with zero change."""
    reward_calc = CompositeReward()
    
    # No entropy change -> no reward
    prev_entropy = 0.5
    current_entropy = 0.5
    reward = reward_calc.calculate_info_gain_reward(prev_entropy, current_entropy)
    assert reward == 0.0


def test_stealth_reward_similar_distributions():
    """Test stealth reward with similar distributions."""
    reward_calc = CompositeReward()
    
    # Similar distributions -> low divergence -> high (less negative) reward
    traffic_dist = torch.tensor([[0.5, 0.3, 0.2]])
    baseline_dist = torch.tensor([[0.5, 0.3, 0.2]])
    
    reward = reward_calc.calculate_stealth_reward(traffic_dist, baseline_dist)
    
    # Should be close to 0 (minimal divergence)
    assert reward.item() < 0.1


def test_stealth_reward_different_distributions():
    """Test stealth reward with different distributions."""
    reward_calc = CompositeReward()
    
    # Different distributions -> high divergence -> low (more negative) reward
    traffic_dist = torch.tensor([[0.9, 0.05, 0.05]])
    baseline_dist = torch.tensor([[0.1, 0.45, 0.45]])
    
    reward = reward_calc.calculate_stealth_reward(traffic_dist, baseline_dist)
    
    # Should be negative (some divergence)
    assert reward.item() < 0


def test_stealth_reward_shape_mismatch():
    """Test stealth reward with mismatched shapes."""
    reward_calc = CompositeReward()
    
    # Mismatched shapes should return 0
    traffic_dist = torch.tensor([[0.5, 0.5]])
    baseline_dist = torch.tensor([[0.3, 0.3, 0.4]])
    
    reward = reward_calc.calculate_stealth_reward(traffic_dist, baseline_dist)
    assert reward.item() == 0.0


def test_composite_reward_compute():
    """Test full composite reward computation."""
    reward_calc = CompositeReward(w_adv=1.0, w_info=0.5, w_stealth=0.3)
    
    adv_score = torch.tensor([0.8])
    info_gain = 0.5
    traffic_dist = torch.tensor([[0.5, 0.3, 0.2]])
    baseline_dist = torch.tensor([[0.5, 0.3, 0.2]])
    
    total_reward = reward_calc.compute(adv_score, info_gain, traffic_dist, baseline_dist)
    
    # Check that result is a tensor
    assert isinstance(total_reward, torch.Tensor)
    
    # Approximate calculation:
    # r_adv = 0.8, r_info = 0.5, r_stealth ≈ 0
    # total ≈ 1.0 * 0.8 + 0.5 * 0.5 + 0.3 * 0 = 1.05
    assert total_reward.item() > 0.5


def test_composite_reward_with_batch():
    """Test composite reward with batched tensors."""
    reward_calc = CompositeReward()
    
    # Batch of adversarial scores
    adv_scores = torch.tensor([0.9, 0.7, 0.5])
    
    # Should handle batch
    rewards = reward_calc.calculate_adversarial_reward(adv_scores)
    assert rewards.shape == (3,)
    assert torch.allclose(rewards, adv_scores)


def test_reward_values_reasonable():
    """Test that reward values are in reasonable ranges."""
    reward_calc = CompositeReward()
    
    # Adversarial rewards should be in [0, 1]
    for score in [0.0, 0.5, 1.0]:
        reward = reward_calc.calculate_adversarial_reward(torch.tensor([score]))
        assert 0 <= reward.item() <= 1
    
    # Info gain rewards should be non-negative
    for prev, curr in [(1.0, 0.5), (0.5, 1.0), (0.5, 0.5)]:
        reward = reward_calc.calculate_info_gain_reward(prev, curr)
        assert reward >= 0
