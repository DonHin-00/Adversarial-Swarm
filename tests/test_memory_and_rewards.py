"""Unit tests for memory components: LogEncoder, SyntheticExperienceGenerator, rewards."""

import torch


class TestLogEncoder:
    """Tests for the graph_store.LogEncoder."""

    def test_empty_logs(self):
        from hive_zero_core.memory.graph_store import LogEncoder

        encoder = LogEncoder(node_feature_dim=64)
        data = encoder.update([])

        assert data.x.shape == (0, 64)
        assert data.edge_index.shape == (2, 0)

    def test_single_log_entry(self):
        from hive_zero_core.memory.graph_store import LogEncoder

        encoder = LogEncoder(node_feature_dim=64)
        logs = [{"src_ip": "192.168.1.1", "dst_ip": "10.0.0.1", "port": 80, "proto": 6}]
        data = encoder.update(logs)

        assert data.x.shape[0] == 2  # 2 nodes
        assert data.x.shape[1] == 64
        assert data.edge_index.shape == (2, 1)  # 1 edge

    def test_invalid_ip_fallback(self):
        from hive_zero_core.memory.graph_store import LogEncoder

        encoder = LogEncoder(node_feature_dim=64)
        logs = [{"src_ip": "invalid", "dst_ip": "10.0.0.1", "port": 80, "proto": 6}]
        data = encoder.update(logs)

        # Should not raise; invalid IP falls back to 0.0.0.0
        assert data.x.shape[0] == 2

    def test_port_clamping(self):
        from hive_zero_core.memory.graph_store import LogEncoder

        encoder = LogEncoder(node_feature_dim=64)
        logs = [{"src_ip": "1.1.1.1", "dst_ip": "2.2.2.2", "port": 99999, "proto": 6}]
        data = encoder.update(logs)

        # Port out of range should be clamped to 0
        assert data.edge_index.shape == (2, 1)

    def test_proto_clamping(self):
        from hive_zero_core.memory.graph_store import LogEncoder

        encoder = LogEncoder(node_feature_dim=64)
        logs = [{"src_ip": "1.1.1.1", "dst_ip": "2.2.2.2", "port": 80, "proto": 999}]
        data = encoder.update(logs)

        # Proto out of range should fall back to TCP (6)
        assert data.edge_index.shape == (2, 1)

    def test_reset_clears_state(self):
        """LogEncoder.reset() should clear the IP mapping for bounded memory."""
        from hive_zero_core.memory.graph_store import LogEncoder

        encoder = LogEncoder(node_feature_dim=64)
        logs = [{"src_ip": "1.1.1.1", "dst_ip": "2.2.2.2", "port": 80, "proto": 6}]
        encoder.update(logs)
        assert encoder.next_idx == 2  # Two IPs registered

        encoder.reset()
        assert encoder.next_idx == 0
        assert len(encoder.ip_to_idx) == 0
        assert len(encoder.idx_to_ip) == 0

    def test_empty_graph_device_consistency(self):
        """Empty-graph returns should be on the module's device."""
        from hive_zero_core.memory.graph_store import LogEncoder

        encoder = LogEncoder(node_feature_dim=64)
        data = encoder.update([])
        # Should be on same device as ip_projection weight (CPU by default)
        assert data.x.device == encoder.ip_projection.weight.device


class TestSyntheticExperienceGenerator:
    """Tests for foundation.SyntheticExperienceGenerator."""

    def test_generate_batch_shapes(self):
        from hive_zero_core.memory.foundation import SyntheticExperienceGenerator

        gen = SyntheticExperienceGenerator(observation_dim=64, action_dim=32)
        obs, acts, rews, next_obs, dones = gen.generate_batch(batch_size=100)

        assert obs.shape == (100, 64)
        assert acts.shape == (100, 32)
        assert rews.shape == (100, 1)
        assert next_obs.shape == (100, 64)
        assert dones.shape == (100, 1)

    def test_rewards_are_positive(self):
        from hive_zero_core.memory.foundation import SyntheticExperienceGenerator

        gen = SyntheticExperienceGenerator(observation_dim=64, action_dim=32)
        _, _, rews, _, _ = gen.generate_batch(batch_size=50)

        assert (rews > 0).all()


class TestCompositeReward:
    """Tests for training.rewards.CompositeReward."""

    def test_info_gain_positive(self):
        from hive_zero_core.training.rewards import CompositeReward

        rc = CompositeReward()
        gain = rc.calculate_info_gain_reward(prev_entropy=0.8, current_entropy=0.5)
        assert gain > 0

    def test_info_gain_clamped_at_zero(self):
        from hive_zero_core.training.rewards import CompositeReward

        rc = CompositeReward()
        gain = rc.calculate_info_gain_reward(prev_entropy=0.3, current_entropy=0.8)
        assert gain == 0.0

    def test_stealth_reward_no_nan(self):
        from hive_zero_core.training.rewards import CompositeReward

        rc = CompositeReward()
        # Use softmax to get valid probability distributions
        traffic = torch.softmax(torch.randn(1, 10), dim=-1)
        baseline = torch.softmax(torch.randn(1, 10), dim=-1)
        reward = rc.calculate_stealth_reward(traffic, baseline)

        assert not torch.isnan(reward)
        assert not torch.isinf(reward)

    def test_stealth_reward_shape_mismatch(self):
        from hive_zero_core.training.rewards import CompositeReward

        rc = CompositeReward()
        traffic = torch.randn(1, 10)
        baseline = torch.randn(1, 5)
        reward = rc.calculate_stealth_reward(traffic, baseline)

        # Shape mismatch should return 0.0 fallback
        assert reward.item() == 0.0

    def test_stealth_reward_with_zeros(self):
        """Verify KL divergence handles zero-valued distributions without NaN."""
        from hive_zero_core.training.rewards import CompositeReward

        rc = CompositeReward()
        traffic = torch.zeros(1, 10)  # All zeros — would cause log(0) without clamping
        baseline = torch.softmax(torch.randn(1, 10), dim=-1)
        reward = rc.calculate_stealth_reward(traffic, baseline)

        assert not torch.isnan(reward)
        assert not torch.isinf(reward)


class TestWeightInitializer:
    """Tests for foundation.WeightInitializer."""

    def test_inject_instincts_runs(self):
        from hive_zero_core.memory.foundation import WeightInitializer
        from hive_zero_core.hive_mind import GatingNetwork

        gating = GatingNetwork(input_dim=64, num_experts=14)
        # Should not raise
        WeightInitializer.inject_instincts(gating)

    def test_defense_bias_elevated(self):
        from hive_zero_core.memory.foundation import WeightInitializer
        from hive_zero_core.hive_mind import GatingNetwork

        gating = GatingNetwork(input_dim=64, num_experts=14)
        WeightInitializer.inject_instincts(gating)

        # Last linear layer bias for indices 10-13 should be elevated
        last_linear = gating.net[-1]
        assert last_linear.bias[10].item() > last_linear.bias[5].item()
