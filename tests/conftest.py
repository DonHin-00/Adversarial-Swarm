"""Test configuration and fixtures for Adversarial-Swarm tests."""

import pytest
import torch


@pytest.fixture
def device():
    """Provide a torch device for tests."""
    return torch.device("cpu")


@pytest.fixture
def observation_dim():
    """Standard observation dimension for tests."""
    return 64


@pytest.fixture
def batch_size():
    """Standard batch size for tests."""
    return 4
