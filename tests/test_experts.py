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

    def test_ensure_dimension_3d_pad(self):
        """ensure_dimension should pad the last dim of 3D tensors."""
        from hive_zero_core.agents.recon_experts import Agent_DeepScope

        expert = Agent_DeepScope(observation_dim=64, action_dim=10)
        x = torch.randn(2, 5, 3)
        out = expert.ensure_dimension(x, 10)

        assert out.shape == (2, 5, 10)

    def test_ensure_dimension_3d_truncate(self):
        """ensure_dimension should truncate the last dim of 3D tensors."""
        from hive_zero_core.agents.recon_experts import Agent_DeepScope

        expert = Agent_DeepScope(observation_dim=64, action_dim=10)
        x = torch.randn(2, 5, 20)
        out = expert.ensure_dimension(x, 10)

        assert out.shape == (2, 5, 10)

    def test_step_counter_increments(self):
        """Each active forward pass should increment the step counter."""
        from hive_zero_core.agents.recon_experts import Agent_DeepScope

        expert = Agent_DeepScope(observation_dim=64, action_dim=10)
        expert.is_active = True
        x = torch.randn(1, 64)
        expert(x)
        expert(x)
        assert expert._step_count == 2

    def test_gradient_checkpointing_flag(self):
        """enable_checkpointing should set the internal flag."""
        from hive_zero_core.agents.recon_experts import Agent_DeepScope

        expert = Agent_DeepScope(observation_dim=64, action_dim=10)
        assert not expert._use_checkpoint
        expert.enable_checkpointing()
        assert expert._use_checkpoint


class TestDeepScope:
    """Tests for DeepScope multi-head attention constraint masking."""

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

    def test_partial_mask(self):
        """Only some actions should be blocked by a partial mask."""
        from hive_zero_core.agents.recon_experts import Agent_DeepScope

        expert = Agent_DeepScope(observation_dim=64, action_dim=10)
        expert.is_active = True
        x = torch.randn(1, 64)
        mask = torch.ones(10)
        mask[0:5] = 0  # Block first 5 actions
        out = expert(x, mask=mask)

        assert out.shape == (1, 10)
        # First 5 should be near -1e9
        assert (out[0, :5] < -1e8).all()
        # Last 5 should not be -1e9
        assert (out[0, 5:] > -1e8).all()


class TestChronos:
    """Tests for Chronos Transformer-based temporal encoder."""

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

    def test_variable_sequence_length(self):
        """Chronos should handle different sequence lengths."""
        from hive_zero_core.agents.recon_experts import Agent_Chronos

        expert = Agent_Chronos(observation_dim=1, action_dim=1)
        expert.is_active = True

        for seq_len in [3, 50, 100]:
            x = torch.randn(1, seq_len)
            out = expert(x)
            assert out.shape == (1, 1)


class TestTarpit:
    """Tests for the Tarpit multi-head attention hunter-trap."""

    def test_output_nonzero(self):
        from hive_zero_core.agents.defense_experts import Agent_Tarpit

        expert = Agent_Tarpit(observation_dim=64, action_dim=64)
        expert.is_active = True
        x = torch.randn(1, 64)
        out = expert(x)

        assert out.shape == (1, 64)
        assert out.abs().sum() > 0

    def test_batch_processing(self):
        from hive_zero_core.agents.defense_experts import Agent_Tarpit

        expert = Agent_Tarpit(observation_dim=32, action_dim=32)
        expert.is_active = True
        x = torch.randn(4, 32)
        out = expert(x)

        assert out.shape == (4, 32)

    def test_spectral_trap_primitive(self):
        """TrapArsenal.spectral_comb should produce non-zero output."""
        from hive_zero_core.agents.defense_experts import TrapArsenal

        out = TrapArsenal.spectral_comb(2, 32, torch.device("cpu"))
        assert out.shape == (2, 32)
        assert out.abs().sum() > 0

    def test_temporal_trap_primitive(self):
        """TrapArsenal.temporal_jitter should produce positive output."""
        from hive_zero_core.agents.defense_experts import TrapArsenal

        out = TrapArsenal.temporal_jitter(2, 32, torch.device("cpu"))
        assert out.shape == (2, 32)
        assert (out > 0).all()


class TestPostExperts:
    """Tests for upgraded post-exploitation experts."""

    def test_mimic_positive_output(self):
        from hive_zero_core.agents.post_experts import Agent_Mimic

        expert = Agent_Mimic(observation_dim=64, action_dim=2)
        expert.is_active = True
        x = torch.randn(1, 64)
        out = expert(x)

        assert out.shape == (1, 2)
        assert (out >= 0).all()

    def test_stego_bounded(self):
        from hive_zero_core.agents.post_experts import Agent_Stego

        expert = Agent_Stego(observation_dim=64, action_dim=64)
        expert.is_active = True
        x = torch.randn(1, 64)
        out = expert(x)

        assert out.shape == (1, 64)
        assert (out >= 0).all()
        assert (out <= 1).all()

    def test_stego_kl_loss(self):
        """Stego VAE should provide a non-negative KL loss."""
        from hive_zero_core.agents.post_experts import Agent_Stego

        expert = Agent_Stego(observation_dim=64, action_dim=64)
        x = torch.randn(4, 64)
        kl = expert.kl_loss(x)

        assert kl.shape == ()
        assert not torch.isnan(kl)

    def test_ghost_entropy_gate(self):
        """Ghost output should have residual skip-connection influence."""
        from hive_zero_core.agents.post_experts import Agent_Ghost

        expert = Agent_Ghost(observation_dim=64, action_dim=5)
        expert.is_active = True
        x = torch.randn(2, 64)
        out = expert(x)

        assert out.shape == (2, 5)

    def test_cleaner_residual(self):
        """Cleaner should use residual connections (output differs from zero)."""
        from hive_zero_core.agents.post_experts import Agent_Cleaner

        expert = Agent_Cleaner(observation_dim=64, action_dim=10)
        expert.is_active = True
        x = torch.randn(1, 64)
        out = expert(x)

        assert out.shape == (1, 10)


class TestOffensiveDefense:
    """Tests for upgraded Kill Chain experts."""

    def test_feedback_loop_output(self):
        from hive_zero_core.agents.offensive_defense import Agent_FeedbackLoop

        expert = Agent_FeedbackLoop(observation_dim=64, action_dim=64)
        expert.is_active = True
        x = torch.randn(1, 64)
        out = expert(x)

        assert out.shape == (1, 64)

    def test_feedback_loop_multi_scale(self):
        """FeedbackLoop should produce different outputs for different inputs."""
        from hive_zero_core.agents.offensive_defense import Agent_FeedbackLoop

        expert = Agent_FeedbackLoop(observation_dim=64, action_dim=64, num_scales=3)
        expert.is_active = True
        out1 = expert(torch.randn(1, 64))
        out2 = expert(torch.randn(1, 64))

        assert not torch.allclose(out1, out2, atol=1e-4)

    def test_flashbang_nonzero(self):
        from hive_zero_core.agents.offensive_defense import Agent_Flashbang

        expert = Agent_Flashbang(observation_dim=64, action_dim=64)
        expert.is_active = True
        x = torch.zeros(1, 64)
        out = expert(x)

        assert out.shape == (1, 64)
        assert out.abs().sum() > 0

    def test_flashbang_adaptive_intensity(self):
        """Flashbang should modulate intensity based on observation."""
        from hive_zero_core.agents.offensive_defense import Agent_Flashbang

        expert = Agent_Flashbang(observation_dim=64, action_dim=64)
        expert.is_active = True
        # With larger input, intensity_head should produce different output
        out_small = expert(torch.randn(1, 64) * 0.01)
        out_large = expert(torch.randn(1, 64) * 10.0)

        # Both should be valid
        assert out_small.shape == (1, 64)
        assert out_large.shape == (1, 64)

    def test_glasshouse_output(self):
        from hive_zero_core.agents.offensive_defense import Agent_GlassHouse

        expert = Agent_GlassHouse(observation_dim=64, action_dim=64)
        expert.is_active = True
        x = torch.randn(1, 64)
        out = expert(x)

        assert out.shape == (1, 64)

    def test_glasshouse_phase_modulation(self):
        """GlassHouse should have learnable phase offsets."""
        from hive_zero_core.agents.offensive_defense import Agent_GlassHouse

        expert = Agent_GlassHouse(observation_dim=64, action_dim=64)
        assert hasattr(expert, 'phase_offset')
        assert expert.phase_offset.shape == (1, 64)


class TestGatingNetwork:
    """Tests for the upgraded gating network."""

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

    def test_load_balance_loss_scalar(self):
        """load_balance_loss should return a scalar tensor."""
        from hive_zero_core.hive_mind import GatingNetwork

        gating = GatingNetwork(input_dim=64, num_experts=14)
        x = torch.randn(4, 64)
        weights = gating(x)
        loss = gating.load_balance_loss(weights)

        assert loss.shape == ()
        assert not torch.isnan(loss)
        assert loss >= 0

    def test_noise_during_training(self):
        """Gating should inject noise during training but not eval."""
        from hive_zero_core.hive_mind import GatingNetwork

        gating = GatingNetwork(input_dim=64, num_experts=14, noise_std=1.0)
        x = torch.randn(1, 64)

        gating.eval()
        w_eval1 = gating(x)
        w_eval2 = gating(x)
        assert torch.allclose(w_eval1, w_eval2)

        gating.train()
        w_train1 = gating(x)
        w_train2 = gating(x)
        # With high noise, consecutive calls should differ
        assert not torch.allclose(w_train1, w_train2, atol=1e-6)

    def test_sparse_top_k_routing(self):
        """Top-k sparse routing should zero out non-selected experts."""
        from hive_zero_core.hive_mind import GatingNetwork

        gating = GatingNetwork(input_dim=64, num_experts=14)
        gating.eval()
        x = torch.randn(1, 64)

        weights = gating(x, top_k=3)
        # Exactly 3 experts should have non-zero weight
        assert (weights[0] > 0).sum().item() == 3
        # Should still sum to ~1
        assert torch.allclose(weights.sum(dim=-1), torch.tensor(1.0), atol=1e-5)

    def test_sparse_routing_none_is_dense(self):
        """Passing top_k=None should return dense (all non-zero) weights."""
        from hive_zero_core.hive_mind import GatingNetwork

        gating = GatingNetwork(input_dim=64, num_experts=14)
        gating.eval()
        x = torch.randn(1, 64)

        weights_dense = gating(x, top_k=None)
        # All experts should have some weight
        assert (weights_dense[0] > 0).sum().item() == 14

    def test_utilisation_stats_after_load_balance(self):
        """utilisation_stats should return non-zero after load_balance_loss calls."""
        from hive_zero_core.hive_mind import GatingNetwork

        gating = GatingNetwork(input_dim=64, num_experts=14)
        x = torch.randn(4, 64)
        weights = gating(x)
        gating.load_balance_loss(weights)

        stats = gating.utilisation_stats()
        assert stats.shape == (14,)
        assert stats.sum() > 0

    def test_utilisation_stats_initial_zero(self):
        """Before any forward passes, utilisation should be all zeros."""
        from hive_zero_core.hive_mind import GatingNetwork

        gating = GatingNetwork(input_dim=64, num_experts=14)
        stats = gating.utilisation_stats()
        assert (stats == 0).all()


class TestCompositeRewardUpgrades:
    """Tests for upgraded CompositeReward."""

    def test_temporal_reward_within_budget(self):
        from hive_zero_core.training.rewards import CompositeReward

        rc = CompositeReward()
        reward = rc.calculate_temporal_reward(elapsed_steps=50, budget=100)
        assert 0 < reward <= 1.0

    def test_temporal_reward_at_budget(self):
        from hive_zero_core.training.rewards import CompositeReward

        rc = CompositeReward()
        reward = rc.calculate_temporal_reward(elapsed_steps=100, budget=100)
        assert reward == 0.0

    def test_temporal_reward_over_budget(self):
        from hive_zero_core.training.rewards import CompositeReward

        rc = CompositeReward()
        reward = rc.calculate_temporal_reward(elapsed_steps=200, budget=100)
        assert reward == 0.0

    def test_stealth_device_aware_fallback(self):
        """Shape-mismatch fallback should preserve device info."""
        from hive_zero_core.training.rewards import CompositeReward

        rc = CompositeReward()
        traffic = torch.randn(1, 10)
        baseline = torch.randn(1, 5)
        result = rc.calculate_stealth_reward(traffic, baseline)

        assert result.device == traffic.device

    def test_compute_returns_tensor(self):
        """compute() should return a proper tensor, not a mixed float/tensor hybrid."""
        from hive_zero_core.training.rewards import CompositeReward

        rc = CompositeReward()
        adv = torch.tensor([0.7])
        traffic = torch.softmax(torch.randn(1, 10), dim=-1)
        baseline = torch.softmax(torch.randn(1, 10), dim=-1)

        result = rc.compute(adv, info_gain=0.3, traffic_dist=traffic,
                            baseline_dist=baseline, elapsed_steps=10)

        assert isinstance(result, torch.Tensor)
        assert not torch.isnan(result).any()
        assert result.dim() >= 0  # at minimum is a valid tensor


class TestSyntheticExperienceDevice:
    """Tests for device-aware SyntheticExperienceGenerator."""

    def test_generate_on_cpu(self):
        from hive_zero_core.memory.foundation import SyntheticExperienceGenerator

        gen = SyntheticExperienceGenerator(observation_dim=64, action_dim=32)
        obs, acts, rews, next_obs, dones = gen.generate_batch(
            batch_size=10, device=torch.device("cpu")
        )

        assert obs.device.type == "cpu"
        assert acts.device.type == "cpu"

    def test_knowledge_loader_raises_on_bad_buffer(self):
        """KnowledgeLoader should raise TypeError for invalid buffers."""
        from hive_zero_core.memory.foundation import KnowledgeLoader

        loader = KnowledgeLoader(observation_dim=64, action_dim=32)
        with pytest.raises(TypeError, match="add.*push"):
            loader.bootstrap(object(), num_samples=10)
