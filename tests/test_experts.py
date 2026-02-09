"""Unit tests for expert agents: base_expert, recon, post, defense, offensive_defense."""

import torch
import pytest


class TestBaseExpert:
    """Tests for base_expert.BaseExpert."""

    def test_inactive_returns_zeros(self):
        from hive_zero_core.agents.recon_experts import Agent_DeepScope

        expert = Agent_DeepScope(observation_dim=64, action_dim=10)
        expert.is_active = False
        x = torch.randn(2, 64)
        out = expert(x)

        assert out.shape == (2, 10)
        assert (out == 0).all()

    def test_active_returns_nonzero(self):
        from hive_zero_core.agents.recon_experts import Agent_DeepScope

        expert = Agent_DeepScope(observation_dim=64, action_dim=10)
        expert.is_active = True
        x = torch.randn(2, 64)
        out = expert(x)

        assert out.shape == (2, 10)

    def test_type_error_on_non_tensor(self):
        from hive_zero_core.agents.recon_experts import Agent_DeepScope

        expert = Agent_DeepScope(observation_dim=64, action_dim=10)
        expert.is_active = True

        with pytest.raises(TypeError):
            expert([1, 2, 3])

    def test_ensure_dimension_pad(self):
        from hive_zero_core.agents.recon_experts import Agent_DeepScope

        expert = Agent_DeepScope(observation_dim=64, action_dim=10)
        x = torch.randn(2, 5)
        out = expert.ensure_dimension(x, 10)

        assert out.shape == (2, 10)

    def test_ensure_dimension_truncate(self):
        from hive_zero_core.agents.recon_experts import Agent_DeepScope

        expert = Agent_DeepScope(observation_dim=64, action_dim=10)
        x = torch.randn(2, 20)
        out = expert.ensure_dimension(x, 10)

        assert out.shape == (2, 10)


class TestDeepScope:
    """Tests for DeepScope mask broadcasting."""

    def test_mask_1d_broadcasts(self):
        from hive_zero_core.agents.recon_experts import Agent_DeepScope

        expert = Agent_DeepScope(observation_dim=64, action_dim=10)
        expert.is_active = True
        x = torch.randn(2, 64)
        mask = torch.ones(10)  # 1D mask
        out = expert(x, mask=mask)

        assert out.shape == (2, 10)
        # With all-ones mask, output should equal adapter output
        assert not torch.all(out == -1e9)

    def test_mask_zeros_blocks(self):
        from hive_zero_core.agents.recon_experts import Agent_DeepScope

        expert = Agent_DeepScope(observation_dim=64, action_dim=10)
        expert.is_active = True
        x = torch.randn(1, 64)
        mask = torch.zeros(10)  # Block everything
        out = expert(x, mask=mask)

        # All outputs should be -1e9 (blocked)
        assert torch.allclose(out, torch.tensor(-1e9).expand_as(out), atol=1.0)

    def test_no_mask_returns_logits(self):
        from hive_zero_core.agents.recon_experts import Agent_DeepScope

        expert = Agent_DeepScope(observation_dim=64, action_dim=10)
        expert.is_active = True
        x = torch.randn(1, 64)
        out = expert(x, mask=None)

        assert out.shape == (1, 10)


class TestChronos:
    """Tests for Chronos time-series expert."""

    def test_forward_2d_input(self):
        from hive_zero_core.agents.recon_experts import Agent_Chronos

        expert = Agent_Chronos(observation_dim=1, action_dim=1)
        expert.is_active = True
        x = torch.randn(2, 10)  # [batch=2, seq_len=10]
        out = expert(x)

        assert out.shape == (2, 1)

    def test_forward_3d_input(self):
        from hive_zero_core.agents.recon_experts import Agent_Chronos

        expert = Agent_Chronos(observation_dim=1, action_dim=1)
        expert.is_active = True
        x = torch.randn(2, 10, 1)  # Already [batch, seq, 1]
        out = expert(x)

        assert out.shape == (2, 1)


class TestTarpit:
    """Tests for the Tarpit defense expert."""

    def test_output_nonzero(self):
        from hive_zero_core.agents.defense_experts import Agent_Tarpit

        expert = Agent_Tarpit(observation_dim=64, action_dim=64)
        expert.is_active = True
        x = torch.randn(1, 64)
        out = expert(x)

        assert out.shape == (1, 64)
        # Output should be non-zero (Tarpit fills all ports)
        assert out.abs().sum() > 0

    def test_batch_processing(self):
        from hive_zero_core.agents.defense_experts import Agent_Tarpit

        expert = Agent_Tarpit(observation_dim=32, action_dim=32)
        expert.is_active = True
        x = torch.randn(4, 32)
        out = expert(x)

        assert out.shape == (4, 32)


class TestPostExperts:
    """Tests for post-exploitation experts."""

    def test_mimic_positive_output(self):
        from hive_zero_core.agents.post_experts import Agent_Mimic

        expert = Agent_Mimic(observation_dim=64, action_dim=2)
        expert.is_active = True
        x = torch.randn(1, 64)
        out = expert(x)

        assert out.shape == (1, 2)
        # Softplus output should be positive
        assert (out >= 0).all()

    def test_stego_bounded(self):
        from hive_zero_core.agents.post_experts import Agent_Stego

        expert = Agent_Stego(observation_dim=64, action_dim=64)
        expert.is_active = True
        x = torch.randn(1, 64)
        out = expert(x)

        assert out.shape == (1, 64)
        # Sigmoid output should be in [0, 1]
        assert (out >= 0).all()
        assert (out <= 1).all()


class TestOffensiveDefense:
    """Tests for Kill Chain experts."""

    def test_feedback_loop_output(self):
        from hive_zero_core.agents.offensive_defense import Agent_FeedbackLoop

        expert = Agent_FeedbackLoop(observation_dim=64, action_dim=64)
        expert.is_active = True
        x = torch.randn(1, 64)
        out = expert(x)

        assert out.shape == (1, 64)

    def test_flashbang_nonzero(self):
        from hive_zero_core.agents.offensive_defense import Agent_Flashbang

        expert = Agent_Flashbang(observation_dim=64, action_dim=64)
        expert.is_active = True
        x = torch.zeros(1, 64)
        out = expert(x)

        assert out.shape == (1, 64)
        # Due to + 1.0 bias, output should never be all zeros
        assert out.abs().sum() > 0

    def test_glasshouse_output(self):
        from hive_zero_core.agents.offensive_defense import Agent_GlassHouse

        expert = Agent_GlassHouse(observation_dim=64, action_dim=64)
        expert.is_active = True
        x = torch.randn(1, 64)
        out = expert(x)

        assert out.shape == (1, 64)


class TestGatingNetwork:
    """Tests for the gating network."""

    def test_softmax_sums_to_one(self):
        from hive_zero_core.hive_mind import GatingNetwork

        gating = GatingNetwork(input_dim=64, num_experts=14)
        x = torch.randn(1, 64)
        weights = gating(x)

        assert weights.shape == (1, 14)
        assert torch.allclose(weights.sum(dim=-1), torch.tensor(1.0), atol=1e-5)

    def test_all_weights_positive(self):
        from hive_zero_core.hive_mind import GatingNetwork

        gating = GatingNetwork(input_dim=64, num_experts=14)
        x = torch.randn(1, 64)
        weights = gating(x)

        assert (weights >= 0).all()
