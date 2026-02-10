"""
Tests for Skill Knowledge and Advanced Reward Systems
"""

import pytest
import torch

from hive_zero_core.agents.base_expert import SkillLevel
from hive_zero_core.agents.recon_experts import CartographerAgent
from hive_zero_core.memory.threat_intelligence import (
    SkillComplexity,
    SkillKnowledge,
    ThreatCategory,
    ThreatIntelligence,
    ThreatIntelligenceStore,
)
from hive_zero_core.training.advanced_rewards import AdvancedCompositeReward


class TestSkillSystem:
    """Test skill proficiency and tracking"""

    def test_skill_levels(self):
        """Test skill level multipliers"""
        agent = CartographerAgent(64, 64)

        assert agent.skill_level == SkillLevel.EXPERT
        assert agent.get_skill_multiplier() == 1.3

    def test_skill_tracking(self):
        """Test skill usage statistics"""
        agent = CartographerAgent(64, 64)

        assert agent.skill_stats["activations"] == 0
        assert agent.skill_stats["successes"] == 0

        # Record successful activation
        agent.record_activation(success=True, confidence=0.8)
        assert agent.skill_stats["activations"] == 1
        assert agent.skill_stats["successes"] == 1
        assert agent.skill_stats["avg_confidence"] == 0.8

        # Record failed activation
        agent.record_activation(success=False, confidence=0.5)
        assert agent.skill_stats["activations"] == 2
        assert agent.skill_stats["failures"] == 1
        assert agent.skill_stats["avg_confidence"] == 0.65

    def test_effectiveness_score(self):
        """Test effectiveness calculation"""
        agent = CartographerAgent(64, 64)

        # Untested agent
        assert agent.get_effectiveness_score() == 0.5

        # After successful uses
        for _ in range(10):
            agent.record_activation(success=True, confidence=0.9)

        effectiveness = agent.get_effectiveness_score()
        assert effectiveness > 0.8  # High success rate + expert level

    def test_primary_skills(self):
        """Test primary and secondary skills"""
        agent = CartographerAgent(64, 64)

        assert "recon_001" in agent.primary_skills
        assert "recon_002" in agent.primary_skills
        assert "recon_003" in agent.secondary_skills


class TestThreatIntelligenceStore:
    """Test knowledge store functionality"""

    def test_initialization(self):
        """Test store initializes with base skills"""
        store = ThreatIntelligenceStore()

        assert len(store.skills) > 0
        assert "recon_001" in store.skills
        assert "attack_001" in store.skills

    def test_get_skill(self):
        """Test skill retrieval"""
        store = ThreatIntelligenceStore()

        skill = store.get_skill("recon_001")
        assert skill is not None
        assert skill.name == "Network Topology Mapping"
        assert skill.category == ThreatCategory.RECONNAISSANCE

    def test_skills_by_category(self):
        """Test filtering skills by category"""
        store = ThreatIntelligenceStore()

        recon_skills = store.get_skills_by_category(ThreatCategory.RECONNAISSANCE)
        assert len(recon_skills) >= 3

        attack_skills = store.get_skills_by_category(ThreatCategory.EXECUTION)
        assert len(attack_skills) >= 1

    def test_skills_by_complexity(self):
        """Test filtering skills by complexity"""
        store = ThreatIntelligenceStore()

        simple_skills = store.get_skills_by_complexity(SkillComplexity.SIMPLE)
        advanced_skills = store.get_skills_by_complexity(SkillComplexity.ADVANCED)

        assert len(simple_skills) >= 1
        assert len(advanced_skills) >= 1

    def test_agent_knowledge_registration(self):
        """Test registering agent knowledge"""
        store = ThreatIntelligenceStore()

        store.register_agent_knowledge("Cartographer", ["recon_001", "recon_002"])
        agent_skills = store.get_agent_skills("Cartographer")

        assert len(agent_skills) == 2
        assert agent_skills[0].skill_id in ["recon_001", "recon_002"]

    def test_knowledge_sharing(self):
        """Test sharing knowledge between agents"""
        store = ThreatIntelligenceStore()

        store.register_agent_knowledge("Agent1", ["recon_001"])
        store.share_knowledge("Agent1", "Agent2", "recon_001")

        agent2_skills = store.get_agent_skills("Agent2")
        assert len(agent2_skills) == 1
        assert agent2_skills[0].skill_id == "recon_001"

    def test_skill_usage_recording(self):
        """Test recording skill usage"""
        store = ThreatIntelligenceStore()

        skill = store.get_skill("recon_001")
        initial_usage = skill.usage_count

        store.record_skill_usage("recon_001", success=True, epoch=1)

        assert skill.usage_count == initial_usage + 1
        assert skill.last_used_epoch == 1

    def test_skill_recommendations(self):
        """Test skill recommendation system"""
        store = ThreatIntelligenceStore()

        # Agent with basic skills
        store.register_agent_knowledge("Novice", ["recon_001", "recon_002"])

        # Should recommend attack_001 since prerequisites are met
        recommendations = store.get_recommended_skills("Novice")
        rec_ids = [r.skill_id for r in recommendations]

        assert "attack_001" in rec_ids  # Prerequisites: recon_001, recon_002

    def test_threat_intel_operations(self):
        """Test threat intelligence management"""
        store = ThreatIntelligenceStore()

        intel = ThreatIntelligence(
            intel_id="test_001",
            threat_type="test_threat",
            description="Test threat",
            indicators=["indicator1"],
            confidence=0.6,
        )

        store.add_threat_intel(intel)

        # Query by confidence
        high_conf = store.get_threat_intel(min_confidence=0.5)
        assert len(high_conf) == 1

        # Validate and update confidence
        store.validate_threat_intel("test_001", is_valid=True)
        assert intel.confidence > 0.6

    def test_knowledge_summary(self):
        """Test knowledge base summary statistics"""
        store = ThreatIntelligenceStore()

        summary = store.get_knowledge_summary()

        assert "total_skills" in summary
        assert "total_intel" in summary
        assert "skills_by_category" in summary
        assert summary["total_skills"] > 0


class TestAdvancedRewards:
    """Test advanced reward system with multiple affect types"""

    def test_initialization(self):
        """Test reward system initialization"""
        reward = AdvancedCompositeReward(
            w_adv=1.0,
            w_info=0.5,
            w_stealth=0.8,
            w_temporal=0.4,
            w_resource=0.3,
            w_reliability=0.6,
            w_novelty=0.5,
            w_coordination=0.7,
        )

        assert reward.w_adv == 1.0
        assert reward.w_temporal == 0.4

    def test_adversarial_reward(self):
        """Test adversarial evasion reward"""
        reward = AdvancedCompositeReward()

        score = torch.tensor([0.85])
        r_adv = reward.calculate_adversarial_reward(score)

        assert r_adv == 0.85

    def test_temporal_reward(self):
        """Test temporal (speed/timing) reward"""
        reward = AdvancedCompositeReward()

        # Faster than target
        r_temporal = reward.calculate_temporal_reward(
            actual_duration=2.0, target_duration=3.0, timing_precision=0.9
        )
        assert r_temporal > 0.8

    def test_resource_reward(self):
        """Test resource efficiency reward"""
        reward = AdvancedCompositeReward()

        # Low resource usage
        r_resource = reward.calculate_resource_reward(compute_cost=0.3, memory_usage=0.2)
        assert r_resource > 0.6

    def test_reliability_reward(self):
        """Test reliability/consistency reward"""
        reward = AdvancedCompositeReward()

        # High success rate
        r_reliability = reward.calculate_reliability_reward(
            success_count=8, total_attempts=10, consistency_score=0.9
        )
        assert r_reliability > 0.7

    def test_novelty_reward(self):
        """Test novelty/discovery reward"""
        reward = AdvancedCompositeReward()

        # Novel action
        r_novelty = reward.calculate_novelty_reward(
            is_novel=True, novelty_score=0.8, exploration_bonus=0.2
        )
        assert r_novelty > 0.8

        # Non-novel action
        r_routine = reward.calculate_novelty_reward(is_novel=False, novelty_score=0.8)
        assert r_routine < r_novelty

    def test_coordination_reward(self):
        """Test multi-agent coordination reward"""
        reward = AdvancedCompositeReward()

        # Optimal team size with good synergy
        r_coord = reward.calculate_coordination_reward(
            active_agents=3, synergy_score=0.8, optimal_team_size=3
        )
        assert r_coord > 0.6

    def test_comprehensive_compute(self):
        """Test comprehensive reward computation"""
        reward = AdvancedCompositeReward()

        rewards = reward.compute(
            adv_score=torch.tensor([0.85]),
            info_gain=0.3,
            traffic_dist=torch.tensor([[0.2, 0.3, 0.5]]),
            baseline_dist=torch.tensor([[0.25, 0.25, 0.5]]),
            actual_duration=2.5,
            target_duration=3.0,
            timing_precision=0.92,
            compute_cost=0.4,
            memory_usage=0.3,
            success_count=8,
            total_attempts=10,
            consistency_score=0.85,
            is_novel=True,
            novelty_score=0.75,
            active_agents=3,
            synergy_score=0.8,
        )

        assert "total" in rewards
        assert "adversarial" in rewards
        assert "temporal" in rewards
        assert "resource" in rewards
        assert "reliability" in rewards
        assert "novelty" in rewards
        assert "coordination" in rewards

        # Total should be sum of weighted components
        assert rewards["total"].item() > 0

    def test_stealth_reward_with_kl(self):
        """Test stealth reward using KL divergence"""
        reward = AdvancedCompositeReward()

        # Identical distributions
        dist1 = torch.tensor([[0.3, 0.3, 0.4]])
        dist2 = torch.tensor([[0.3, 0.3, 0.4]])

        r_stealth = reward.calculate_stealth_reward(dist1, dist2)
        assert r_stealth > -0.1  # Close to 0 KL divergence

        # Very different distributions
        dist3 = torch.tensor([[0.1, 0.1, 0.8]])
        r_stealth_diff = reward.calculate_stealth_reward(dist1, dist3)
        assert r_stealth_diff < r_stealth  # Higher KL = lower reward

    def test_shape_mismatch_handling(self):
        """Test handling of mismatched tensor shapes"""
        reward = AdvancedCompositeReward()

        dist1 = torch.tensor([[0.3, 0.3, 0.4]])
        dist2 = torch.tensor([[0.5, 0.5]])  # Different shape

        r_stealth = reward.calculate_stealth_reward(dist1, dist2)
        assert r_stealth == 0.0  # Should return 0 for mismatched shapes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
