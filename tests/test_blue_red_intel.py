"""Unit tests for blue team, red booster, and threat intel database."""

import torch
import pytest


# ======================================================================
# Blue Team Detection Stack
# ======================================================================

class TestWAF:
    """Tests for the adversarial WAF expert."""

    def test_output_shape(self):
        from hive_zero_core.agents.blue_team import Agent_WAF

        waf = Agent_WAF(observation_dim=64, action_dim=2)
        waf.is_active = True
        x = torch.randn(2, 64)
        out = waf(x)

        assert out.shape == (2, 2)

    def test_signature_bank_shape(self):
        from hive_zero_core.agents.blue_team import Agent_WAF

        waf = Agent_WAF(observation_dim=64, num_signatures=32)
        assert waf.signature_bank.shape == (32, 128)  # hidden_dim default=128

    def test_update_signatures(self):
        from hive_zero_core.agents.blue_team import Agent_WAF

        waf = Agent_WAF(observation_dim=64, num_signatures=16)
        original = waf.signature_bank.clone()
        waf.update_signatures(torch.randn(5, 128))
        # At least some signatures should have changed
        assert not torch.allclose(original, waf.signature_bank)

    def test_inactive_returns_zeros(self):
        from hive_zero_core.agents.blue_team import Agent_WAF

        waf = Agent_WAF(observation_dim=64, action_dim=2)
        waf.is_active = False
        out = waf(torch.randn(1, 64))
        assert (out == 0).all()


class TestEDR:
    """Tests for the EDR behavioural detector."""

    def test_output_shape_2d(self):
        from hive_zero_core.agents.blue_team import Agent_EDR

        edr = Agent_EDR(observation_dim=64, action_dim=2)
        edr.is_active = True
        out = edr(torch.randn(2, 64))
        assert out.shape == (2, 2)

    def test_output_shape_3d(self):
        from hive_zero_core.agents.blue_team import Agent_EDR

        edr = Agent_EDR(observation_dim=64, action_dim=2)
        edr.is_active = True
        out = edr(torch.randn(2, 10, 64))  # Sequence input
        assert out.shape == (2, 2)


class TestSIEM:
    """Tests for the SIEM log correlator."""

    def test_output_shape(self):
        from hive_zero_core.agents.blue_team import Agent_SIEM

        siem = Agent_SIEM(observation_dim=64, action_dim=2)
        siem.is_active = True
        out = siem(torch.randn(3, 64))
        assert out.shape == (3, 2)

    def test_alert_prototypes_exist(self):
        from hive_zero_core.agents.blue_team import Agent_SIEM

        siem = Agent_SIEM(observation_dim=64, num_alert_prototypes=16)
        assert siem.alert_prototypes.shape == (1, 16, 128)


class TestIDS:
    """Tests for the IDS deep-packet inspector."""

    def test_output_shape(self):
        from hive_zero_core.agents.blue_team import Agent_IDS

        ids_agent = Agent_IDS(observation_dim=64, action_dim=2)
        ids_agent.is_active = True
        out = ids_agent(torch.randn(2, 64))
        assert out.shape == (2, 2)

    def test_multi_scale_convolutions(self):
        """IDS should have three conv layers with different kernel sizes."""
        from hive_zero_core.agents.blue_team import Agent_IDS

        ids_agent = Agent_IDS(observation_dim=64)
        assert ids_agent.conv3.kernel_size == (3,)
        assert ids_agent.conv5.kernel_size == (5,)
        assert ids_agent.conv7.kernel_size == (7,)


# ======================================================================
# Red Team Pre-Attack Booster
# ======================================================================

class TestPreAttackBooster:
    """Tests for the adversarial payload booster."""

    def test_output_shape(self):
        from hive_zero_core.agents.red_booster import Agent_PreAttackBooster

        booster = Agent_PreAttackBooster(observation_dim=64, action_dim=64)
        booster.is_active = True
        out = booster(torch.randn(2, 64))
        assert out.shape == (2, 64)

    def test_different_obs_action_dims(self):
        from hive_zero_core.agents.red_booster import Agent_PreAttackBooster

        booster = Agent_PreAttackBooster(observation_dim=64, action_dim=128)
        booster.is_active = True
        out = booster(torch.randn(1, 64))
        assert out.shape == (1, 128)

    def test_register_blue_team(self):
        from hive_zero_core.agents.red_booster import Agent_PreAttackBooster
        from hive_zero_core.agents.blue_team import Agent_WAF

        booster = Agent_PreAttackBooster(observation_dim=64, action_dim=64)
        waf = Agent_WAF(observation_dim=64)
        booster.register_blue_team([waf])
        assert len(booster._blue_detectors) == 1

    def test_without_blue_team(self):
        """Booster should work even without registered detectors."""
        from hive_zero_core.agents.red_booster import Agent_PreAttackBooster

        booster = Agent_PreAttackBooster(observation_dim=64, action_dim=64)
        booster.is_active = True
        out = booster(torch.randn(1, 64))
        assert out.shape == (1, 64)
        assert not torch.isnan(out).any()

    def test_gate_is_bounded(self):
        """Gate values should be in [0, 1] (sigmoid output)."""
        from hive_zero_core.agents.red_booster import Agent_PreAttackBooster

        booster = Agent_PreAttackBooster(observation_dim=64, action_dim=64)
        x = torch.randn(4, 64)
        evasion = booster.evasion_encoder(x)
        gate_input = torch.cat([x, evasion], dim=-1)
        g = booster.gate(gate_input)
        assert (g >= 0).all() and (g <= 1).all()


# ======================================================================
# Threat Intelligence Database
# ======================================================================

class TestThreatIntelDB:
    """Tests for the evolving threat intelligence database."""

    def test_initial_state(self):
        from hive_zero_core.memory.threat_intel_db import ThreatIntelDB

        db = ThreatIntelDB(embedding_dim=64, bank_size=32)
        assert db.attack_bank.shape == (32, 64)
        assert db.defense_bank.shape == (32, 64)
        assert db.generation.item() == 0

    def test_record_attack_success(self):
        from hive_zero_core.memory.threat_intel_db import ThreatIntelDB

        db = ThreatIntelDB(embedding_dim=64, bank_size=32)
        original = db.attack_bank.clone()
        db.record_attack_success(torch.randn(5, 64))
        # Bank should have changed
        assert not torch.allclose(original, db.attack_bank)

    def test_record_attack_failure(self):
        from hive_zero_core.memory.threat_intel_db import ThreatIntelDB

        db = ThreatIntelDB(embedding_dim=64, bank_size=32)
        original = db.defense_bank.clone()
        db.record_attack_failure(torch.randn(5, 64))
        assert not torch.allclose(original, db.defense_bank)

    def test_step_generation(self):
        from hive_zero_core.memory.threat_intel_db import ThreatIntelDB

        db = ThreatIntelDB(embedding_dim=64)
        db.step_generation(evasion_rate=0.75)
        assert db.generation.item() == 1
        db.step_generation(evasion_rate=0.80)
        assert db.generation.item() == 2

    def test_query_attack_bank(self):
        from hive_zero_core.memory.threat_intel_db import ThreatIntelDB

        db = ThreatIntelDB(embedding_dim=64, bank_size=16)
        query = torch.randn(2, 64)
        sigs, sims = db.query_attack_bank(query, top_k=3)

        assert sigs.shape == (2, 3, 64)
        assert sims.shape == (2, 3)

    def test_query_defense_bank(self):
        from hive_zero_core.memory.threat_intel_db import ThreatIntelDB

        db = ThreatIntelDB(embedding_dim=64, bank_size=16)
        query = torch.randn(1, 64)
        sigs, sims = db.query_defense_bank(query, top_k=5)

        assert sigs.shape == (1, 5, 64)
        assert sims.shape == (1, 5)

    def test_novelty_score(self):
        from hive_zero_core.memory.threat_intel_db import ThreatIntelDB

        db = ThreatIntelDB(embedding_dim=64, bank_size=16)
        embs = torch.randn(4, 64)
        scores = db.novelty_score(embs)

        assert scores.shape == (4,)
        # Novelty should be between 0 and 2 (cosine distance range)
        assert (scores >= 0).all()
        assert not torch.isnan(scores).any()

    def test_get_stats(self):
        from hive_zero_core.memory.threat_intel_db import ThreatIntelDB

        db = ThreatIntelDB(embedding_dim=64)
        db.step_generation(0.5)
        db.step_generation(0.7)

        stats = db.get_stats()
        assert stats["generation"] == 2
        assert "avg_evasion_rate" in stats
        assert "evasion_trend" in stats

    def test_ema_update_converges(self):
        """Repeated ingestion of the same embedding should push the
        nearest signature towards that embedding."""
        from hive_zero_core.memory.threat_intel_db import ThreatIntelDB

        torch.manual_seed(42)
        db = ThreatIntelDB(embedding_dim=16, bank_size=4, ema_decay=0.9,
                           novelty_threshold=0.5)
        target = torch.ones(1, 16) * 5.0

        for _ in range(50):
            db.record_attack_success(target)

        # The closest signature should be very close to the target
        _, sims = db.query_attack_bank(target, top_k=1)
        assert sims[0, 0].item() > 0.95  # High cosine similarity

    def test_novelty_prevents_duplicates(self):
        """Very similar embeddings should EMA-merge rather than fill the bank."""
        from hive_zero_core.memory.threat_intel_db import ThreatIntelDB

        torch.manual_seed(42)
        db = ThreatIntelDB(embedding_dim=16, bank_size=8, novelty_threshold=0.05)
        base = torch.randn(1, 16)

        for _ in range(20):
            # Tiny perturbations — should merge, not allocate new slots
            db.record_attack_success(base + torch.randn(1, 16) * 0.001)

        stats = db.get_stats()
        # Should not have filled all 8 slots with near-identical entries
        assert stats["attack_bank_fill"] < 8

    def test_warmup_novelty_threshold(self):
        """During warmup, the effective novelty threshold should be higher (more permissive)."""
        from hive_zero_core.memory.threat_intel_db import ThreatIntelDB

        db = ThreatIntelDB(embedding_dim=16, bank_size=8,
                           novelty_threshold=0.1, novelty_warmup_gens=100)

        # At generation 0, effective threshold should be 1.0 (accept everything)
        assert db._effective_novelty_threshold() == 1.0

        # Advance halfway through warmup
        for _ in range(50):
            db.step_generation(0.5)
        thresh_mid = db._effective_novelty_threshold()
        assert 0.1 < thresh_mid < 1.0

        # After warmup completes, threshold should equal configured value
        for _ in range(50):
            db.step_generation(0.5)
        assert abs(db._effective_novelty_threshold() - 0.1) < 1e-6

    def test_warmup_default_50_gens(self):
        """Default warmup period is 50 generations."""
        from hive_zero_core.memory.threat_intel_db import ThreatIntelDB

        db = ThreatIntelDB(embedding_dim=16)
        assert db.novelty_warmup_gens == 50


# ======================================================================
# Integration: GatingNetwork with new expert count
# ======================================================================

class TestGatingNetworkWithNewExperts:
    """Verify gating network handles the expanded expert roster."""

    def test_19_experts_softmax(self):
        from hive_zero_core.hive_mind import GatingNetwork

        gating = GatingNetwork(input_dim=64, num_experts=19)
        weights = gating(torch.randn(1, 64))
        assert weights.shape == (1, 19)
        assert torch.allclose(weights.sum(dim=-1), torch.tensor(1.0), atol=1e-5)
