"""
MITRE ATT&CK and ATLAS Query Tools

Provides a high-level interface for querying, searching, and analysing the MITRE
technique database.  The :class:`MITREQueryTool` wraps the raw technique dictionaries
from :mod:`mitre_mapping` and adds coverage analysis, recommendation, and reporting
functionality needed by the broader Adversarial-Swarm system.
"""

from typing import Dict, List, Optional

from hive_zero_core.mitre.mitre_mapping import (
    MITRE_ATLAS_TECHNIQUES,
    MITRE_ATTACK_TECHNIQUES,
    MITRETechnique,
    get_all_techniques,
    get_technique,
    get_techniques_by_tactic,
)


class MITREQueryTool:
    """
    High-level query interface for the MITRE ATT&CK and ATLAS technique databases.

    Provides methods for:
    - Looking up individual techniques by ID or name
    - Filtering techniques by tactic, platform, or keyword
    - Analysing coverage of a set of capability IDs against the database
    - Recommending unmapped techniques for a given tactic
    - Generating summary reports of technique coverage
    """

    def __init__(self) -> None:
        self._all: Dict[str, MITRETechnique] = get_all_techniques()

    # ------------------------------------------------------------------
    # Single-technique lookups
    # ------------------------------------------------------------------

    def get_technique(self, technique_id: str) -> Optional[MITRETechnique]:
        """Return the :class:`MITRETechnique` for *technique_id*, or ``None``."""
        return get_technique(technique_id)

    def get_techniques_by_tactic(
        self, tactic: str, include_atlas: bool = True
    ) -> List[MITRETechnique]:
        """Return all techniques belonging to *tactic*.

        Args:
            tactic: Tactic value string (e.g. ``"TA0043"`` or ``"AML.TA0001"``).
            include_atlas: When ``True`` (default) also include ATLAS techniques.
        """
        return get_techniques_by_tactic(tactic, include_atlas=include_atlas)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_by_name(self, keyword: str, case_sensitive: bool = False) -> List[MITRETechnique]:
        """Return all techniques whose *name* contains *keyword*.

        Args:
            keyword: Substring to search for.
            case_sensitive: When ``False`` (default) the comparison is case-insensitive.
        """
        if not case_sensitive:
            keyword = keyword.lower()
            return [t for t in self._all.values() if keyword in t.name.lower()]
        return [t for t in self._all.values() if keyword in t.name]

    def search_by_description(
        self, keyword: str, case_sensitive: bool = False
    ) -> List[MITRETechnique]:
        """Return all techniques whose *description* contains *keyword*."""
        if not case_sensitive:
            keyword = keyword.lower()
            return [t for t in self._all.values() if keyword in t.description.lower()]
        return [t for t in self._all.values() if keyword in t.description]

    def search_by_platform(self, platform: str) -> List[MITRETechnique]:
        """Return all techniques applicable to *platform* (e.g. ``"Windows"``).

        The match is case-insensitive substring check against each technique's
        ``platform`` list.
        """
        platform_lower = platform.lower()
        return [
            t
            for t in self._all.values()
            if any(platform_lower in p.lower() for p in t.platform)
        ]

    # ------------------------------------------------------------------
    # Coverage analysis
    # ------------------------------------------------------------------

    def get_coverage_report(
        self, mapped_technique_ids: List[str]
    ) -> Dict[str, object]:
        """Analyse how many MITRE techniques are covered by *mapped_technique_ids*.

        Args:
            mapped_technique_ids: List of technique IDs that are already mapped to
                system capabilities.

        Returns:
            A dictionary with keys:
            - ``"total_techniques"``: int — total techniques in the database.
            - ``"covered"``: int — techniques present in *mapped_technique_ids*.
            - ``"uncovered"``: int — techniques not yet mapped.
            - ``"coverage_pct"``: float — percentage covered (0–100).
            - ``"covered_ids"``: List[str] — IDs that are mapped.
            - ``"uncovered_ids"``: List[str] — IDs not yet mapped.
            - ``"by_tactic"``: Dict[str, Dict] — per-tactic breakdown.
        """
        mapped_set = set(mapped_technique_ids)
        all_ids = set(self._all.keys())

        covered_ids = sorted(all_ids & mapped_set)
        uncovered_ids = sorted(all_ids - mapped_set)

        total = len(all_ids)
        covered = len(covered_ids)
        coverage_pct = (covered / total * 100) if total > 0 else 0.0

        # Per-tactic breakdown
        tactic_breakdown: Dict[str, Dict] = {}
        for tech in self._all.values():
            tactic = tech.tactic
            if tactic not in tactic_breakdown:
                tactic_breakdown[tactic] = {"total": 0, "covered": 0, "ids": []}
            tactic_breakdown[tactic]["total"] += 1
            tactic_breakdown[tactic]["ids"].append(tech.technique_id)
            if tech.technique_id in mapped_set:
                tactic_breakdown[tactic]["covered"] += 1

        return {
            "total_techniques": total,
            "covered": covered,
            "uncovered": total - covered,
            "coverage_pct": round(coverage_pct, 2),
            "covered_ids": covered_ids,
            "uncovered_ids": uncovered_ids,
            "by_tactic": tactic_breakdown,
        }

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def recommend_techniques(
        self,
        tactic: Optional[str] = None,
        is_atlas: Optional[bool] = None,
        limit: int = 10,
    ) -> List[MITRETechnique]:
        """Return up to *limit* technique recommendations.

        Args:
            tactic: If provided, restrict recommendations to this tactic value string.
            is_atlas: If ``True``, only ATLAS techniques; if ``False``, only ATT&CK;
                      ``None`` (default) includes both.
            limit: Maximum number of techniques to return.
        """
        candidates = list(self._all.values())

        if tactic is not None:
            candidates = [t for t in candidates if t.tactic == tactic]

        if is_atlas is True:
            candidates = [t for t in candidates if t.is_atlas]
        elif is_atlas is False:
            candidates = [t for t in candidates if not t.is_atlas]

        return candidates[:limit]

    # ------------------------------------------------------------------
    # Summary reports
    # ------------------------------------------------------------------

    def get_summary(self) -> Dict[str, object]:
        """Return a high-level summary of the technique database.

        Returns a dictionary with:
        - ``"attack_count"``: Number of ATT&CK Enterprise techniques.
        - ``"atlas_count"``: Number of ATLAS techniques.
        - ``"total_count"``: Total techniques in the combined database.
        - ``"tactic_distribution"``: Dict mapping tactic → count.
        """
        tactic_dist: Dict[str, int] = {}
        for tech in self._all.values():
            tactic_dist[tech.tactic] = tactic_dist.get(tech.tactic, 0) + 1

        return {
            "attack_count": len(MITRE_ATTACK_TECHNIQUES),
            "atlas_count": len(MITRE_ATLAS_TECHNIQUES),
            "total_count": len(self._all),
            "tactic_distribution": tactic_dist,
        }

    def get_atlas_summary(self) -> Dict[str, object]:
        """Return a summary of only the MITRE ATLAS (AI/ML) techniques."""
        tactic_dist: Dict[str, int] = {}
        for tech in MITRE_ATLAS_TECHNIQUES.values():
            tactic_dist[tech.tactic] = tactic_dist.get(tech.tactic, 0) + 1

        return {
            "atlas_count": len(MITRE_ATLAS_TECHNIQUES),
            "tactic_distribution": tactic_dist,
        }
