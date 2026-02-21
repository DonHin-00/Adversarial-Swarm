"""Unit tests for hive_zero_core.mitre: technique database and MITREQueryTool."""

import pytest


# ---------------------------------------------------------------------------
# mitre_mapping module tests
# ---------------------------------------------------------------------------


class TestMITREMapping:
    """Tests for the raw technique dictionaries and helper functions."""

    def test_attack_techniques_populated(self):
        from hive_zero_core.mitre.mitre_mapping import MITRE_ATTACK_TECHNIQUES

        assert len(MITRE_ATTACK_TECHNIQUES) >= 100, (
            f"Expected ≥100 ATT&CK techniques, got {len(MITRE_ATTACK_TECHNIQUES)}"
        )

    def test_atlas_techniques_populated(self):
        from hive_zero_core.mitre.mitre_mapping import MITRE_ATLAS_TECHNIQUES

        assert len(MITRE_ATLAS_TECHNIQUES) >= 40, (
            f"Expected ≥40 ATLAS techniques, got {len(MITRE_ATLAS_TECHNIQUES)}"
        )

    def test_total_techniques_count(self):
        from hive_zero_core.mitre.mitre_mapping import get_all_techniques

        total = len(get_all_techniques())
        assert total >= 145, f"Expected ≥145 total techniques, got {total}"

    def test_get_technique_attack(self):
        from hive_zero_core.mitre.mitre_mapping import get_technique

        t = get_technique("T1059")
        assert t is not None
        assert t.technique_id == "T1059"
        assert t.name == "Command and Scripting Interpreter"
        assert not t.is_atlas

    def test_get_technique_atlas(self):
        from hive_zero_core.mitre.mitre_mapping import get_technique

        t = get_technique("AML.T0043")
        assert t is not None
        assert t.technique_id == "AML.T0043"
        assert t.is_atlas

    def test_get_technique_missing_returns_none(self):
        from hive_zero_core.mitre.mitre_mapping import get_technique

        assert get_technique("T9999") is None
        assert get_technique("AML.T9999") is None

    def test_get_techniques_by_tactic(self):
        from hive_zero_core.mitre.mitre_mapping import MITRETactic, get_techniques_by_tactic

        techniques = get_techniques_by_tactic(MITRETactic.RECONNAISSANCE.value, include_atlas=False)
        assert len(techniques) >= 2
        assert all(t.tactic == MITRETactic.RECONNAISSANCE.value for t in techniques)

    def test_get_all_techniques_includes_both(self):
        from hive_zero_core.mitre.mitre_mapping import (
            MITRE_ATLAS_TECHNIQUES,
            MITRE_ATTACK_TECHNIQUES,
            get_all_techniques,
        )

        all_tech = get_all_techniques()
        # Should contain at least one known ATT&CK and one known ATLAS technique
        assert "T1059" in all_tech
        assert "AML.T0043" in all_tech
        assert len(all_tech) == len(MITRE_ATTACK_TECHNIQUES) + len(MITRE_ATLAS_TECHNIQUES)

    def test_technique_dataclass_fields(self):
        from hive_zero_core.mitre.mitre_mapping import get_technique

        t = get_technique("T1055")
        assert hasattr(t, "technique_id")
        assert hasattr(t, "name")
        assert hasattr(t, "description")
        assert hasattr(t, "tactic")
        assert hasattr(t, "platform")
        assert hasattr(t, "data_sources")
        assert hasattr(t, "is_atlas")

    def test_new_attack_v18_supply_chain(self):
        """Supply-chain techniques added in ATT&CK v18 should be present."""
        from hive_zero_core.mitre.mitre_mapping import get_technique

        t = get_technique("T1195")
        assert t is not None
        assert "Supply Chain" in t.name

    def test_new_attack_v18_container(self):
        """Container escape technique (T1611) added in ATT&CK v18 should be present."""
        from hive_zero_core.mitre.mitre_mapping import get_technique

        t = get_technique("T1611")
        assert t is not None
        assert "Escape" in t.name

    def test_new_atlas_2026_agentic(self):
        """Agentic AI ATLAS 2026 techniques should be present."""
        from hive_zero_core.mitre.mitre_mapping import get_technique

        for tech_id in ("AML.T0097", "AML.T0103", "AML.T0108", "AML.T0109"):
            t = get_technique(tech_id)
            assert t is not None, f"Expected {tech_id} to be in the database"
            assert t.is_atlas


# ---------------------------------------------------------------------------
# MITREQueryTool tests
# ---------------------------------------------------------------------------


class TestMITREQueryTool:
    """Tests for the MITREQueryTool high-level interface."""

    @pytest.fixture
    def tool(self):
        from hive_zero_core.mitre.query_tools import MITREQueryTool

        return MITREQueryTool()

    def test_get_technique_known(self, tool):
        t = tool.get_technique("T1059")
        assert t is not None
        assert t.technique_id == "T1059"

    def test_get_technique_unknown_returns_none(self, tool):
        assert tool.get_technique("T0000") is None

    def test_get_techniques_by_tactic(self, tool):
        from hive_zero_core.mitre.mitre_mapping import MITRETactic

        results = tool.get_techniques_by_tactic(MITRETactic.IMPACT.value, include_atlas=False)
        assert len(results) >= 3
        assert all(t.tactic == MITRETactic.IMPACT.value for t in results)

    def test_search_by_name_case_insensitive(self, tool):
        results = tool.search_by_name("injection")
        assert len(results) >= 1
        assert all("injection" in t.name.lower() for t in results)

    def test_search_by_name_case_sensitive(self, tool):
        # "PowerShell" only matches with capital P and S when case_sensitive=True
        results_exact = tool.search_by_name("PowerShell", case_sensitive=True)
        results_lower = tool.search_by_name("powershell", case_sensitive=True)
        assert len(results_exact) >= 1
        assert len(results_lower) == 0

    def test_search_by_name_no_match(self, tool):
        results = tool.search_by_name("zzznomatchzzz")
        assert results == []

    def test_search_by_description(self, tool):
        results = tool.search_by_description("injection")
        assert len(results) >= 1

    def test_search_by_platform_windows(self, tool):
        results = tool.search_by_platform("Windows")
        assert len(results) >= 10

    def test_search_by_platform_ml_system(self, tool):
        results = tool.search_by_platform("ML System")
        assert len(results) >= 10

    def test_get_coverage_report_full(self, tool):
        from hive_zero_core.mitre.mitre_mapping import get_all_techniques

        all_ids = list(get_all_techniques().keys())
        report = tool.get_coverage_report(all_ids)

        assert report["total_techniques"] == len(all_ids)
        assert report["covered"] == len(all_ids)
        assert report["uncovered"] == 0
        assert report["coverage_pct"] == 100.0

    def test_get_coverage_report_empty(self, tool):
        report = tool.get_coverage_report([])

        assert report["covered"] == 0
        assert report["coverage_pct"] == 0.0
        assert len(report["uncovered_ids"]) == report["total_techniques"]

    def test_get_coverage_report_partial(self, tool):
        mapped = ["T1059", "T1055", "AML.T0043"]
        report = tool.get_coverage_report(mapped)

        assert report["covered"] == 3
        assert set(report["covered_ids"]) == {"T1059", "T1055", "AML.T0043"}
        assert 0 < report["coverage_pct"] < 100

    def test_get_coverage_report_by_tactic_structure(self, tool):
        report = tool.get_coverage_report(["T1059"])
        by_tactic = report["by_tactic"]
        assert isinstance(by_tactic, dict)
        # Each value should have "total", "covered", "ids"
        for tactic_data in by_tactic.values():
            assert "total" in tactic_data
            assert "covered" in tactic_data
            assert "ids" in tactic_data

    def test_recommend_techniques_default(self, tool):
        results = tool.recommend_techniques()
        assert len(results) <= 10

    def test_recommend_techniques_atlas_only(self, tool):
        results = tool.recommend_techniques(is_atlas=True, limit=5)
        assert all(t.is_atlas for t in results)
        assert len(results) <= 5

    def test_recommend_techniques_attack_only(self, tool):
        results = tool.recommend_techniques(is_atlas=False, limit=5)
        assert all(not t.is_atlas for t in results)

    def test_recommend_techniques_with_tactic(self, tool):
        from hive_zero_core.mitre.mitre_mapping import MITRETactic

        results = tool.recommend_techniques(tactic=MITRETactic.PERSISTENCE.value, limit=20)
        assert all(t.tactic == MITRETactic.PERSISTENCE.value for t in results)

    def test_get_summary(self, tool):
        summary = tool.get_summary()

        assert "attack_count" in summary
        assert "atlas_count" in summary
        assert "total_count" in summary
        assert "tactic_distribution" in summary
        assert summary["attack_count"] >= 100
        assert summary["atlas_count"] >= 40
        assert summary["total_count"] == summary["attack_count"] + summary["atlas_count"]

    def test_get_atlas_summary(self, tool):
        summary = tool.get_atlas_summary()

        assert "atlas_count" in summary
        assert "tactic_distribution" in summary
        assert summary["atlas_count"] >= 40

    def test_module_init_exports(self):
        """MITREQueryTool should be importable from the mitre package."""
        from hive_zero_core.mitre import MITREQueryTool

        tool = MITREQueryTool()
        assert tool.get_technique("T1059") is not None
