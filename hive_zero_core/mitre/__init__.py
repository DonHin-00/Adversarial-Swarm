"""
MITRE ATT&CK and MITRE ATLAS Integration

This module provides comprehensive MITRE framework coverage for
the Adversarial-Swarm system.

Includes:
- MITRE ATT&CK Enterprise (200+ techniques)
- MITRE ATLAS (AI/ML adversarial techniques)
- Technique-to-Capability mappings
- MITREQueryTool for lookups, recommendations, and coverage analysis
"""

from .mitre_mapping import (
    MITRETactic,
    MITREATLASTactic,
    MITRETechnique,
    MITRE_ATTACK_TECHNIQUES,
    MITRE_ATLAS_TECHNIQUES,
    get_technique,
    get_techniques_by_tactic,
    get_all_techniques,
)
from .query_tools import MITREQueryTool

__all__ = [
    "MITRETactic",
    "MITREATLASTactic",
    "MITRETechnique",
    "MITRE_ATTACK_TECHNIQUES",
    "MITRE_ATLAS_TECHNIQUES",
    "get_technique",
    "get_techniques_by_tactic",
    "get_all_techniques",
    "MITREQueryTool",
]
