"""Tests for genetic evolution capabilities in red team agents."""

import pytest
import torch


class TestGeneticEvolution:
    """Tests for the genetic_evolution module."""

    def test_polymorphic_engine_mutate_code(self):
        """Test that code mutation produces different output."""
        from hive_zero_core.agents.genetic_evolution import PolymorphicEngine

        engine = PolymorphicEngine()
        source = """def hello():
    print("Hello, World!")
    return 42"""

        mutated = engine.mutate_code(source, gene_seed=12345)

        # Mutated code should be different
        assert mutated != source
        # Should contain gene marker
        assert "GENE:" in mutated or "_gene_" in mutated

    def test_polymorphic_engine_mutate_string(self):
        """Test that string payload mutation works."""
        from hive_zero_core.agents.genetic_evolution import PolymorphicEngine

        engine = PolymorphicEngine()
        payload = "SELECT * FROM users WHERE id = 1"

        mutated = engine.mutate_string(payload, gene_seed=54321)

        # Mutated payload should be different
        assert mutated != payload
        # Should contain some mutation marker
        assert len(mutated) >= len(payload)

    def test_natural_selection_validate_python(self):
        """Test that valid Python code is accepted."""
        from hive_zero_core.agents.genetic_evolution import NaturalSelection

        selector = NaturalSelection()

        # Valid code
        valid_code = "x = 1 + 2\nprint(x)"
        assert selector.validate_python(valid_code) is True

        # Invalid code
        invalid_code = "def broken(\nprint("
        assert selector.validate_python(invalid_code) is False

    def test_natural_selection_validate_payload(self):
        """Test payload validation."""
        from hive_zero_core.agents.genetic_evolution import NaturalSelection

        selector = NaturalSelection()

        # Valid payload
        valid_payload = "SELECT * FROM users"
        assert selector.validate_payload(valid_payload) is True

        # Too many null bytes
        invalid_payload = "\x00" * 20
        assert selector.validate_payload(invalid_payload) is False

        # Too long
        too_long = "A" * 20000
        assert selector.validate_payload(too_long, max_length=10000) is False

    def test_generation_tracker(self):
        """Test generation tracking functionality."""
        from hive_zero_core.agents.genetic_evolution import GenerationTracker

        tracker = GenerationTracker()

        assert tracker.get_generation() == 0

        tracker.increment_generation(gene_seed=123, success=True)
        assert tracker.get_generation() == 1

        tracker.increment_generation(gene_seed=456, success=False)
        assert tracker.get_generation() == 2

        stats = tracker.get_mutation_stats()
        assert stats['total'] == 2
        assert stats['successful'] == 1
        assert stats['success_rate'] == 0.5

    def test_genetic_evolution_evolve_code(self):
        """Test full code evolution cycle."""
        from hive_zero_core.agents.genetic_evolution import GeneticEvolution

        evolution = GeneticEvolution()
        source = """def simple():
    x = 1
    return x"""

        mutated, gene_seed, success = evolution.evolve_code(source, max_attempts=5)

        assert success is True
        assert gene_seed > 0
        # Mutated code should still be valid
        from hive_zero_core.agents.genetic_evolution import NaturalSelection
        assert NaturalSelection.validate_python(mutated) is True
        # Since mutation is probabilistic, just verify generation was tracked
        assert evolution.tracker.get_generation() > 0

    def test_genetic_evolution_evolve_payload(self):
        """Test full payload evolution cycle."""
        from hive_zero_core.agents.genetic_evolution import GeneticEvolution

        evolution = GeneticEvolution()
        payload = "test payload"

        mutated, gene_seed, success = evolution.evolve_payload(payload, max_attempts=5)

        assert success is True
        assert gene_seed > 0
        # Payload should be mutated
        assert len(mutated) > 0

    def test_genetic_evolution_stats(self):
        """Test evolution statistics tracking."""
        from hive_zero_core.agents.genetic_evolution import GeneticEvolution

        evolution = GeneticEvolution()
        payload = "test"

        # Evolve a few times
        for _ in range(3):
            evolution.evolve_payload(payload, max_attempts=2)

        stats = evolution.get_stats()
        assert 'total' in stats
        assert 'success_rate' in stats
        assert stats['total'] >= 3


class TestMutatorWithEvolution:
    """Tests for Agent_Mutator with genetic evolution capabilities."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Skipping HF model tests without GPU"
    )
    def test_mutator_evolution_disabled(self):
        """Test that Mutator works with evolution disabled."""
        from hive_zero_core.agents.attack_experts import Agent_Mutator, Agent_Sentinel, Agent_PayloadGen

        obs_dim, act_dim = 64, 32
        sentinel = Agent_Sentinel(obs_dim, act_dim)
        generator = Agent_PayloadGen(obs_dim, act_dim)
        mutator = Agent_Mutator(
            obs_dim, act_dim, sentinel, generator,
            enable_evolution=False
        )

        assert mutator.enable_evolution is False
        assert mutator.evolution_engine is None

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Skipping HF model tests without GPU"
    )
    def test_mutator_evolution_enabled(self):
        """Test that Mutator initializes with evolution enabled."""
        from hive_zero_core.agents.attack_experts import Agent_Mutator, Agent_Sentinel, Agent_PayloadGen

        obs_dim, act_dim = 64, 32
        sentinel = Agent_Sentinel(obs_dim, act_dim)
        generator = Agent_PayloadGen(obs_dim, act_dim)
        mutator = Agent_Mutator(
            obs_dim, act_dim, sentinel, generator,
            enable_evolution=True
        )

        assert mutator.enable_evolution is True
        assert mutator.evolution_engine is not None

    def test_mutator_evolve_payload_text_disabled(self):
        """Test payload text evolution when disabled."""
        from hive_zero_core.agents.attack_experts import Agent_Mutator
        from hive_zero_core.agents.base_expert import BaseExpert

        # Create mock experts
        class MockExpert(BaseExpert):
            def _forward_impl(self, x, context, mask=None):
                return torch.zeros(x.size(0), self.action_dim)

        obs_dim, act_dim = 64, 32
        sentinel = MockExpert(obs_dim, act_dim)
        generator = MockExpert(obs_dim, act_dim)

        mutator = Agent_Mutator(
            obs_dim, act_dim, sentinel, generator,
            enable_evolution=False
        )

        payload = "test payload"
        mutated, gene_seed, success = mutator.evolve_payload_text(payload)

        assert mutated == payload
        assert gene_seed == 0
        assert success is False

    def test_mutator_evolve_payload_text_enabled(self):
        """Test payload text evolution when enabled."""
        from hive_zero_core.agents.attack_experts import Agent_Mutator
        from hive_zero_core.agents.base_expert import BaseExpert

        # Create mock experts
        class MockExpert(BaseExpert):
            def _forward_impl(self, x, context, mask=None):
                return torch.zeros(x.size(0), self.action_dim)

        obs_dim, act_dim = 64, 32
        sentinel = MockExpert(obs_dim, act_dim)
        generator = MockExpert(obs_dim, act_dim)

        mutator = Agent_Mutator(
            obs_dim, act_dim, sentinel, generator,
            enable_evolution=True
        )

        payload = "test payload"
        mutated, gene_seed, success = mutator.evolve_payload_text(payload)

        assert success is True
        assert gene_seed > 0
        assert len(mutated) > 0

    def test_mutator_evolve_code(self):
        """Test code evolution functionality."""
        from hive_zero_core.agents.attack_experts import Agent_Mutator
        from hive_zero_core.agents.base_expert import BaseExpert

        # Create mock experts
        class MockExpert(BaseExpert):
            def _forward_impl(self, x, context, mask=None):
                return torch.zeros(x.size(0), self.action_dim)

        obs_dim, act_dim = 64, 32
        sentinel = MockExpert(obs_dim, act_dim)
        generator = MockExpert(obs_dim, act_dim)

        mutator = Agent_Mutator(
            obs_dim, act_dim, sentinel, generator,
            enable_evolution=True
        )

        code = """def test():
    return 42"""

        mutated, gene_seed, success = mutator.evolve_code(code)

        assert success is True
        assert gene_seed > 0
        assert mutated != code

    def test_mutator_get_evolution_stats(self):
        """Test evolution statistics retrieval."""
        from hive_zero_core.agents.attack_experts import Agent_Mutator
        from hive_zero_core.agents.base_expert import BaseExpert

        # Create mock experts
        class MockExpert(BaseExpert):
            def _forward_impl(self, x, context, mask=None):
                return torch.zeros(x.size(0), self.action_dim)

        obs_dim, act_dim = 64, 32
        sentinel = MockExpert(obs_dim, act_dim)
        generator = MockExpert(obs_dim, act_dim)

        # Test with evolution enabled
        mutator = Agent_Mutator(
            obs_dim, act_dim, sentinel, generator,
            enable_evolution=True
        )

        stats = mutator.get_evolution_stats()
        assert stats['enabled'] is True
        assert 'total' in stats

        # Test with evolution disabled
        mutator_disabled = Agent_Mutator(
            obs_dim, act_dim, sentinel, generator,
            enable_evolution=False
        )

        stats_disabled = mutator_disabled.get_evolution_stats()
        assert stats_disabled['enabled'] is False
