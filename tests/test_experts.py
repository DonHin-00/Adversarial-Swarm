"""
Tests for individual expert implementations.
These tests check basic structure and initialization without requiring model downloads.
"""

import pytest
import torch

from hive_zero_core.agents.base_expert import BaseExpert
from hive_zero_core.agents.defense_experts import Agent_Tarpit
from hive_zero_core.agents.offensive_defense import (
    Agent_FeedbackLoop,
    Agent_Flashbang,
    Agent_GlassHouse,
)
from hive_zero_core.agents.post_experts import (
    Agent_Cleaner,
    Agent_Ghost,
    Agent_Mimic,
    Agent_Stego,
)
from hive_zero_core.agents.recon_experts import Agent_Cartographer, Agent_Chronos, Agent_DeepScope


def test_base_expert_interface():
    """Test that BaseExpert defines the expected interface."""
    # BaseExpert is abstract, but we can check its methods exist
    assert hasattr(BaseExpert, 'forward')
    assert hasattr(BaseExpert, '_forward_impl')
    assert hasattr(BaseExpert, 'ensure_dimension')


def test_cartographer_initialization():
    """Test Agent_Cartographer initialization."""
    expert = Agent_Cartographer(
        observation_dim=64,
        action_dim=128,
        hidden_dim=64
    )
    
    assert expert.name == "Cartographer"
    assert expert.observation_dim == 64
    assert expert.action_dim == 128


def test_cartographer_forward_without_graph():
    """Test Agent_Cartographer forward pass without graph structure."""
    expert = Agent_Cartographer(
        observation_dim=64,
        action_dim=128,
        hidden_dim=64
    )
    
    # Without edge index, should return zeros
    x = torch.randn(10, 64)
    output = expert(x, context=None)
    
    assert output.shape == (10, 128)
    assert torch.allclose(output, torch.zeros_like(output))


def test_deepscope_initialization():
    """Test Agent_DeepScope initialization."""
    expert = Agent_DeepScope(
        observation_dim=64,
        action_dim=128,
        hidden_dim=64
    )
    
    assert expert.name == "DeepScope"
    assert expert.observation_dim == 64
    assert expert.action_dim == 128


def test_deepscope_forward_without_mask():
    """Test Agent_DeepScope forward pass without mask."""
    expert = Agent_DeepScope(
        observation_dim=64,
        action_dim=128,
        hidden_dim=64
    )
    
    x = torch.randn(5, 64)
    output = expert(x, context=None, mask=None)
    
    assert output.shape == (5, 128)


def test_deepscope_forward_with_mask():
    """Test Agent_DeepScope forward pass with mask."""
    expert = Agent_DeepScope(
        observation_dim=64,
        action_dim=128,
        hidden_dim=64
    )
    
    x = torch.randn(5, 64)
    # Mask: 1 for valid, 0 for invalid
    mask = torch.ones(5, 128)
    mask[:, :64] = 0  # Block first half of actions
    
    output = expert(x, context=None, mask=mask)
    
    assert output.shape == (5, 128)
    # Invalid actions should have very negative values
    assert (output[:, :64] < -1e8).all()


def test_chronos_initialization():
    """Test Agent_Chronos initialization."""
    expert = Agent_Chronos(
        observation_dim=64,
        action_dim=128,
        hidden_dim=64
    )
    
    assert expert.name == "Chronos"
    assert expert.observation_dim == 64
    assert expert.action_dim == 128


def test_chronos_forward():
    """Test Agent_Chronos forward pass."""
    expert = Agent_Chronos(
        observation_dim=64,
        action_dim=128,
        hidden_dim=64
    )
    
    # Time series input: [batch, seq_len]
    x = torch.randn(5, 10)
    output = expert(x, context=None)
    
    assert output.shape == (5, 128)


def test_mimic_initialization():
    """Test Agent_Mimic initialization."""
    expert = Agent_Mimic(
        observation_dim=64,
        action_dim=128,
        hidden_dim=64
    )
    
    assert expert.name == "Mimic"
    assert expert.observation_dim == 64
    assert expert.action_dim == 128


def test_mimic_forward():
    """Test Agent_Mimic forward pass."""
    expert = Agent_Mimic(
        observation_dim=64,
        action_dim=128,
        hidden_dim=64
    )
    
    x = torch.randn(5, 64)
    output = expert(x, context=None)
    
    assert output.shape == (5, 128)
    # Mimic uses softplus, so outputs should be positive
    assert (output >= 0).all()


def test_ghost_initialization():
    """Test Agent_Ghost initialization."""
    expert = Agent_Ghost(
        observation_dim=64,
        action_dim=128,
        hidden_dim=64
    )
    
    assert expert.name == "Ghost"


def test_stego_initialization():
    """Test Agent_Stego initialization."""
    expert = Agent_Stego(
        observation_dim=64,
        action_dim=128,
        hidden_dim=64
    )
    
    assert expert.name == "Stego"


def test_stego_forward():
    """Test Agent_Stego forward pass."""
    expert = Agent_Stego(
        observation_dim=64,
        action_dim=128,
        hidden_dim=64
    )
    
    x = torch.randn(5, 64)
    output = expert(x, context=None)
    
    assert output.shape == (5, 128)
    # Stego uses sigmoid, so outputs should be in [0, 1]
    assert (output >= 0).all() and (output <= 1).all()


def test_cleaner_initialization():
    """Test Agent_Cleaner initialization."""
    expert = Agent_Cleaner(
        observation_dim=64,
        action_dim=128,
        hidden_dim=64
    )
    
    assert expert.name == "Cleaner"


def test_tarpit_initialization():
    """Test Agent_Tarpit initialization."""
    expert = Agent_Tarpit(
        observation_dim=64,
        action_dim=128,
        hidden_dim=128
    )
    
    assert expert.name == "Tarpit"
    assert expert.num_traps == 20


def test_tarpit_forward():
    """Test Agent_Tarpit forward pass."""
    expert = Agent_Tarpit(
        observation_dim=64,
        action_dim=128,
        hidden_dim=128
    )
    
    x = torch.randn(5, 64)
    output = expert(x, context=None)
    
    assert output.shape == (5, 128)
    # Tarpit should produce non-zero outputs
    assert output.abs().sum() > 0


def test_feedbackloop_initialization():
    """Test Agent_FeedbackLoop initialization."""
    expert = Agent_FeedbackLoop(
        observation_dim=64,
        action_dim=128,
        hidden_dim=128
    )
    
    assert expert.name == "FeedbackLoop"


def test_flashbang_initialization():
    """Test Agent_Flashbang initialization."""
    expert = Agent_Flashbang(
        observation_dim=64,
        action_dim=128,
        hidden_dim=128
    )
    
    assert expert.name == "Flashbang"


def test_glasshouse_initialization():
    """Test Agent_GlassHouse initialization."""
    expert = Agent_GlassHouse(
        observation_dim=64,
        action_dim=128,
        hidden_dim=128
    )
    
    assert expert.name == "GlassHouse"


def test_glasshouse_forward():
    """Test Agent_GlassHouse forward pass."""
    expert = Agent_GlassHouse(
        observation_dim=64,
        action_dim=128,
        hidden_dim=128
    )
    
    x = torch.randn(5, 64)
    output = expert(x, context=None)
    
    assert output.shape == (5, 128)
    # GlassHouse should produce positive "exposure" values
    assert (output >= 0).all()


def test_expert_device_handling():
    """Test that experts can be moved to different devices."""
    expert = Agent_Mimic(
        observation_dim=64,
        action_dim=128,
        hidden_dim=64
    )
    
    # Should work on CPU
    x = torch.randn(2, 64)
    output = expert(x, context=None)
    assert output.device.type == 'cpu'


def test_expert_training_mode():
    """Test that experts can switch between train and eval modes."""
    expert = Agent_Ghost(
        observation_dim=64,
        action_dim=128,
        hidden_dim=64
    )
    
    # Default should be training mode
    assert expert.training
    
    # Switch to eval
    expert.eval()
    assert not expert.training
    
    # Switch back to train
    expert.train()
    assert expert.training
