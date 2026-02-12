"""
Capability Escalation System for Swarm Fusion

Implements emergent abilities where merged units gain exponentially more powerful
capabilities. The more merges, the more capabilities unlock.
"""

import logging
import random
from typing import List, Dict, Set, Optional
from enum import Enum
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


class CapabilityTier(Enum):
    """Capability tiers unlocked by merge count."""
    BASIC = 0           # 0 merges - individual agent
    ENHANCED = 1        # 1-2 merges - coordinated agents
    ADVANCED = 2        # 3-5 merges - team tactics
    ELITE = 3           # 6-10 merges - coordinated operations
    EXPERT = 4          # 11-20 merges - advanced techniques
    MASTER = 5          # 21+ merges - cutting-edge methods


@dataclass
class Capability:
    """Represents a specific capability or ability."""
    name: str
    description: str
    tier: CapabilityTier
    power_multiplier: float
    unlock_threshold: int  # Number of merges required
    prerequisites: List[str] = field(default_factory=list)
    synergy_bonus: float = 1.0

    def __repr__(self):
        return f"Capability({self.name}, tier={self.tier.name}, power={self.power_multiplier:.2f}x)"


class CapabilityRegistry:
    """
    Registry of all available capabilities across tiers.
    Defines the capability tree and unlock requirements.
    """

    def __init__(self):
        """Initialize capability registry with full capability tree."""
        self.capabilities: Dict[str, Capability] = {}
        self._initialize_capabilities()

    def _initialize_capabilities(self):
        """Define all capabilities across tiers."""
        
        # === TIER 0: BASIC (Individual) ===
        self._register(Capability(
            "basic_mutation",
            "String substitution and variable renaming for signature variation",
            CapabilityTier.BASIC,
            power_multiplier=1.0,
            unlock_threshold=0
        ))

        self._register(Capability(
            "basic_obfuscation",
            "Control flow flattening and identifier mangling",
            CapabilityTier.BASIC,
            power_multiplier=1.0,
            unlock_threshold=0
        ))

        # === TIER 1: ENHANCED (Duo Power) ===
        self._register(Capability(
            "polymorphic_encoding",
            "Runtime code generation with variable encryption keys",
            CapabilityTier.ENHANCED,
            power_multiplier=1.5,
            unlock_threshold=1,
            prerequisites=["basic_mutation"]
        ))

        self._register(Capability(
            "multi_stage_execution",
            "Staged payload delivery with independent execution contexts",
            CapabilityTier.ENHANCED,
            power_multiplier=1.8,
            unlock_threshold=1
        ))

        self._register(Capability(
            "timing_evasion",
            "Execution delay patterns to evade behavior analysis",
            CapabilityTier.ENHANCED,
            power_multiplier=1.6,
            unlock_threshold=2,
            synergy_bonus=1.2
        ))

        # === TIER 2: ADVANCED (Squad Tactics) ===
        self._register(Capability(
            "parallel_exploitation",
            "Concurrent execution across multiple vulnerability vectors",
            CapabilityTier.ADVANCED,
            power_multiplier=2.5,
            unlock_threshold=3,
            prerequisites=["multi_stage_execution"]
        ))

        self._register(Capability(
            "metamorphic_recompilation",
            "On-the-fly code rewriting with semantic preservation",
            CapabilityTier.ADVANCED,
            power_multiplier=2.2,
            unlock_threshold=4,
            prerequisites=["polymorphic_encoding"]
        ))

        self._register(Capability(
            "distributed_coordination",
            "IPC-based multi-process orchestration with shared state",
            CapabilityTier.ADVANCED,
            power_multiplier=2.8,
            unlock_threshold=5,
            synergy_bonus=1.5
        ))

        self._register(Capability(
            "environment_detection",
            "VM, debugger, and sandbox detection with evasion logic",
            CapabilityTier.ADVANCED,
            power_multiplier=2.0,
            unlock_threshold=4
        ))

        # === TIER 3: ELITE (Strike Force) ===
        self._register(Capability(
            "consensus_protocol",
            "Byzantine fault-tolerant coordination between agents",
            CapabilityTier.ELITE,
            power_multiplier=4.0,
            unlock_threshold=6,
            prerequisites=["distributed_coordination"],
            synergy_bonus=2.0
        ))

        self._register(Capability(
            "behavior_learning",
            "Runtime adaptation based on defense system responses",
            CapabilityTier.ELITE,
            power_multiplier=3.5,
            unlock_threshold=7,
            prerequisites=["metamorphic_recompilation"]
        ))

        self._register(Capability(
            "vulnerability_chaining",
            "Automated exploit chain synthesis from primitive components",
            CapabilityTier.ELITE,
            power_multiplier=4.5,
            unlock_threshold=8,
            prerequisites=["parallel_exploitation", "behavior_learning"]
        ))

        self._register(Capability(
            "traffic_mimicry",
            "Statistical modeling of legitimate traffic patterns",
            CapabilityTier.ELITE,
            power_multiplier=3.8,
            unlock_threshold=9,
            synergy_bonus=1.8
        ))

        # === TIER 4: EXPERT (Advanced Techniques) ===
        self._register(Capability(
            "process_hollowing",
            "Process injection via PE image replacement in suspended processes",
            CapabilityTier.EXPERT,
            power_multiplier=6.0,
            unlock_threshold=11,
            prerequisites=["consensus_protocol"],
            synergy_bonus=3.0
        ))

        self._register(Capability(
            "token_impersonation",
            "Access token duplication and impersonation for privilege escalation",
            CapabilityTier.EXPERT,
            power_multiplier=5.0,
            unlock_threshold=12,
            prerequisites=["behavior_learning"]
        ))

        self._register(Capability(
            "driver_exploitation",
            "Vulnerable driver abuse for kernel-level access (e.g., signed drivers)",
            CapabilityTier.EXPERT,
            power_multiplier=7.0,
            unlock_threshold=15,
            prerequisites=["vulnerability_chaining", "traffic_mimicry"]
        ))

        self._register(Capability(
            "dll_sideloading",
            "DLL search order hijacking for code execution via legitimate binaries",
            CapabilityTier.EXPERT,
            power_multiplier=6.5,
            unlock_threshold=14,
            synergy_bonus=2.5
        ))

        # === TIER 5: MASTER (Cutting-Edge Real Techniques) ===
        self._register(Capability(
            "com_hijacking",
            "COM object hijacking for persistence and privilege escalation",
            CapabilityTier.MASTER,
            power_multiplier=10.0,
            unlock_threshold=21,
            prerequisites=["process_hollowing", "driver_exploitation"],
            synergy_bonus=5.0
        ))

        self._register(Capability(
            "memory_only_execution",
            "Fileless execution with reflective DLL injection and in-memory PE loading",
            CapabilityTier.MASTER,
            power_multiplier=12.0,
            unlock_threshold=25,
            prerequisites=["dll_sideloading", "token_impersonation"]
        ))

        self._register(Capability(
            "bootkit_persistence",
            "MBR/VBR infection for pre-OS persistence (requires physical/admin access)",
            CapabilityTier.MASTER,
            power_multiplier=15.0,
            unlock_threshold=30,
            prerequisites=["com_hijacking", "memory_only_execution"],
            synergy_bonus=10.0
        ))

    def _register(self, capability: Capability):
        """Register a capability in the registry."""
        self.capabilities[capability.name] = capability

    def get_capability(self, name: str) -> Optional[Capability]:
        """Get capability by name."""
        return self.capabilities.get(name)

    def get_unlockable_at_tier(self, tier: CapabilityTier) -> List[Capability]:
        """Get all capabilities at a specific tier."""
        return [cap for cap in self.capabilities.values() if cap.tier == tier]

    def get_capabilities_by_merge_count(self, merge_count: int) -> List[Capability]:
        """Get all capabilities unlockable at this merge count."""
        return [cap for cap in self.capabilities.values()
                if cap.unlock_threshold <= merge_count]


class CapabilityManager:
    """
    Manages capabilities for swarm units.
    Tracks unlocked abilities and calculates power scaling.
    """

    def __init__(self):
        """Initialize capability manager."""
        self.registry = CapabilityRegistry()
        self.unit_capabilities: Dict[str, Set[str]] = {}  # unit_id -> set of capability names
        self.unlock_history: List[Dict] = []

    def unlock_capabilities_for_unit(self, unit_id: str, merge_count: int,
                                     existing_capabilities: Optional[Set[str]] = None) -> List[Capability]:
        """
        Unlock new capabilities based on merge count.

        Args:
            unit_id: ID of the swarm unit
            merge_count: Number of merges performed
            existing_capabilities: Capabilities already unlocked

        Returns:
            List of newly unlocked capabilities
        """
        if unit_id not in self.unit_capabilities:
            self.unit_capabilities[unit_id] = set()

        if existing_capabilities:
            self.unit_capabilities[unit_id].update(existing_capabilities)

        # Get capabilities available at this merge count
        available = self.registry.get_capabilities_by_merge_count(merge_count)

        newly_unlocked = []

        for cap in available:
            # Check if not already unlocked
            if cap.name in self.unit_capabilities[unit_id]:
                continue

            # Check prerequisites
            if all(prereq in self.unit_capabilities[unit_id] for prereq in cap.prerequisites):
                self.unit_capabilities[unit_id].add(cap.name)
                newly_unlocked.append(cap)

                # Record unlock
                self.unlock_history.append({
                    'unit_id': unit_id,
                    'capability': cap.name,
                    'merge_count': merge_count,
                    'tier': cap.tier.name
                })

                logger.info(f"🔓 Unit {unit_id[:8]} unlocked: {cap.name} (tier {cap.tier.name})")

        return newly_unlocked

    def get_unit_capabilities(self, unit_id: str) -> List[Capability]:
        """Get all capabilities for a unit."""
        if unit_id not in self.unit_capabilities:
            return []

        return [self.registry.get_capability(name)
                for name in self.unit_capabilities[unit_id]
                if self.registry.get_capability(name)]

    def calculate_power_multiplier(self, unit_id: str) -> float:
        """
        Calculate total power multiplier from all unlocked capabilities.

        Args:
            unit_id: ID of the swarm unit

        Returns:
            Total power multiplier
        """
        capabilities = self.get_unit_capabilities(unit_id)

        if not capabilities:
            return 1.0

        # Base multiplier from all capabilities
        base_multiplier = sum(cap.power_multiplier for cap in capabilities)

        # Synergy bonuses (multiplicative)
        synergy_multiplier = 1.0
        for cap in capabilities:
            if cap.synergy_bonus > 1.0:
                synergy_multiplier *= cap.synergy_bonus

        # Tier escalation bonus (higher tiers get exponential boost)
        tier_bonus = 1.0
        for cap in capabilities:
            if cap.tier.value >= CapabilityTier.ELITE.value:
                tier_bonus += cap.tier.value * 0.5

        total_multiplier = base_multiplier * synergy_multiplier * tier_bonus

        logger.debug(f"Power multiplier for {unit_id[:8]}: {total_multiplier:.2f}x "
                    f"(base={base_multiplier:.2f}, synergy={synergy_multiplier:.2f}, tier={tier_bonus:.2f})")

        return total_multiplier

    def get_tier_for_merge_count(self, merge_count: int) -> CapabilityTier:
        """Determine capability tier based on merge count."""
        if merge_count >= 21:
            return CapabilityTier.MASTER
        elif merge_count >= 11:
            return CapabilityTier.EXPERT
        elif merge_count >= 6:
            return CapabilityTier.ELITE
        elif merge_count >= 3:
            return CapabilityTier.ADVANCED
        elif merge_count >= 1:
            return CapabilityTier.ENHANCED
        else:
            return CapabilityTier.BASIC

    def describe_capabilities(self, unit_id: str) -> str:
        """Generate human-readable description of unit's capabilities."""
        capabilities = self.get_unit_capabilities(unit_id)

        if not capabilities:
            return "No capabilities unlocked"

        # Group by tier
        by_tier: Dict[CapabilityTier, List[Capability]] = {}
        for cap in capabilities:
            if cap.tier not in by_tier:
                by_tier[cap.tier] = []
            by_tier[cap.tier].append(cap)

        lines = [f"Unit {unit_id[:12]} Capabilities ({len(capabilities)} total):"]

        for tier in sorted(by_tier.keys(), key=lambda t: t.value):
            tier_caps = by_tier[tier]
            lines.append(f"\n  [{tier.name}]")
            for cap in tier_caps:
                lines.append(f"    • {cap.name} ({cap.power_multiplier:.1f}x)")

        power_mult = self.calculate_power_multiplier(unit_id)
        lines.append(f"\n  TOTAL POWER: {power_mult:.2f}x")

        return '\n'.join(lines)

    def get_statistics(self) -> Dict:
        """Get statistics about capability unlocks."""
        if not self.unlock_history:
            return {'total_unlocks': 0}

        tier_counts = {}
        for record in self.unlock_history:
            tier = record['tier']
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        return {
            'total_unlocks': len(self.unlock_history),
            'unique_units': len(self.unit_capabilities),
            'tier_distribution': tier_counts,
            'avg_capabilities_per_unit': (
                sum(len(caps) for caps in self.unit_capabilities.values()) /
                len(self.unit_capabilities) if self.unit_capabilities else 0
            )
        }


class EmergentBehaviors:
    """
    Defines emergent behaviors that appear at higher capability tiers.
    These are special abilities that only emerge from sufficient merging.
    """

    @staticmethod
    def check_emergent_behavior(capabilities: List[Capability]) -> List[str]:
        """
        Check if emergent behaviors have appeared.

        Args:
            capabilities: List of unlocked capabilities

        Returns:
            List of emergent behavior names
        """
        emergent = []
        cap_names = {cap.name for cap in capabilities}

        # Coordinated Operations: Appears when multiple distributed capabilities combine
        distributed_caps = {'consensus_protocol', 'process_hollowing', 'distributed_coordination'}
        if len(cap_names & distributed_caps) >= 2:
            emergent.append("coordinated_operations")

        # Deep Persistence: Appears with system-level persistence + evasion
        persistence_caps = {'timing_evasion', 'traffic_mimicry', 'driver_exploitation'}
        if len(cap_names & persistence_caps) >= 2:
            emergent.append("deep_persistence")

        # Living Off The Land: Appears with native tools + token abuse
        if 'dll_sideloading' in cap_names and 'token_impersonation' in cap_names:
            emergent.append("living_off_the_land")

        # Advanced Threat Actor: Appears with all master-tier capabilities
        apex_caps = {'com_hijacking', 'memory_only_execution', 'bootkit_persistence'}
        if apex_caps.issubset(cap_names):
            emergent.append("advanced_threat_actor")

        # Full Arsenal: Maximum capability density
        if len(capabilities) >= 20:
            emergent.append("full_arsenal")

        return emergent
