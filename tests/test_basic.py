"""Basic tests for the Adversarial-Swarm package."""

import pytest
import torch


def test_import_hive_zero_core():
    """Test that the main package can be imported."""
    import hive_zero_core

    assert hive_zero_core is not None


def test_import_hive_mind():
    """Test that HiveMind can be imported."""
    from hive_zero_core.hive_mind import HiveMind

    assert HiveMind is not None


def test_hive_mind_initialization(observation_dim):
    """Test that HiveMind can be initialized."""
    from hive_zero_core.hive_mind import HiveMind

    hive = HiveMind(observation_dim=observation_dim, pretrained=False)
    assert hive is not None
    assert hive.observation_dim == observation_dim
    assert len(hive.experts) == 14


def test_hive_mind_forward_pass(observation_dim):
    """Test that HiveMind forward pass works."""
    from hive_zero_core.hive_mind import HiveMind

    hive = HiveMind(observation_dim=observation_dim, pretrained=False)

    # Create dummy log data
    raw_logs = [
        {"timestamp": "2024-01-01", "event": "connection", "source_ip": "192.168.1.1"},
        {"timestamp": "2024-01-01", "event": "request", "source_ip": "192.168.1.2"},
    ]

    # Run forward pass
    results = hive.forward(raw_logs, top_k=3)

    # Check that we get results
    assert isinstance(results, dict)
    assert len(results) > 0


def test_gating_network(observation_dim):
    """Test that the gating network works."""
    from hive_zero_core.hive_mind import GatingNetwork

    num_experts = 14
    gating = GatingNetwork(input_dim=observation_dim, num_experts=num_experts)

    # Create dummy input
    x = torch.randn(1, observation_dim)

    # Get weights
    weights = gating(x)

    # Check shape and properties
    assert weights.shape == (1, num_experts)
    assert torch.allclose(weights.sum(dim=-1), torch.tensor(1.0), atol=1e-5)
    assert (weights >= 0).all()
    assert (weights <= 1).all()
