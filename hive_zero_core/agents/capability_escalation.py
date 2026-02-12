"""
Capability Escalation System for Swarm Fusion

Implements emergent abilities where merged units gain exponentially more powerful
capabilities. The more merges, the more capabilities unlock.
"""

import logging
from hive_zero_core.security import SecureRandom, InputValidator, AuditLogger, AccessController
from hive_zero_core.security.audit_logger import SecurityEvent
from hive_zero_core.security.access_control import OperationType

import numpy as np
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
    mitre_attack_id: Optional[str] = None  # MITRE ATT&CK technique ID (e.g., "T1055.012")

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
        # Basic Reconnaissance Variants
        self._register(Capability(
            "icmp_sweep",
            "ICMP ping sweep for host discovery with TTL analysis",
            CapabilityTier.BASIC,
            power_multiplier=1.0,
            unlock_threshold=0
        ))

        self._register(Capability(
            "tcp_syn_scan",
            "TCP SYN scanning with response pattern learning",
            CapabilityTier.BASIC,
            power_multiplier=1.0,
            unlock_threshold=0
        ))

        self._register(Capability(
            "dns_enumeration",
            "DNS queries with zone transfer attempts and subdomain discovery",
            CapabilityTier.BASIC,
            power_multiplier=1.0,
            unlock_threshold=0
        ))

        # Basic Honeypot/Trap Variants
        self._register(Capability(
            "port_listener",
            "TCP port listener with connection fingerprinting and logging to central hub",
            CapabilityTier.BASIC,
            power_multiplier=1.0,
            unlock_threshold=0
        ))

        self._register(Capability(
            "connection_delay",
            "Adaptive tarpit that learns optimal delay patterns from attacker behavior",
            CapabilityTier.BASIC,
            power_multiplier=1.0,
            unlock_threshold=0
        ))

        self._register(Capability(
            "fake_banner",
            "Dynamic service banner that adapts based on reconnaissance feedback",
            CapabilityTier.BASIC,
            power_multiplier=1.0,
            unlock_threshold=0
        ))

        # Red Team Foundations (REINFORCED)
        self._register(Capability(
            "basic_mutation",
            "String substitution with pattern learning from successful evasions",
            CapabilityTier.BASIC,
            power_multiplier=1.0,
            unlock_threshold=0
        ))

        self._register(Capability(
            "basic_obfuscation",
            "Control flow flattening that evolves based on detection feedback",
            CapabilityTier.BASIC,
            power_multiplier=1.0,
            unlock_threshold=0
        ))

        self._register(Capability(
            "intelligence_gathering",
            "Central intelligence hub that aggregates all recon data for growth",
            CapabilityTier.BASIC,
            power_multiplier=1.0,
            unlock_threshold=0
        ))

        # === TIER 1: ENHANCED (Coordinated Agents) ===
        # Enhanced Reconnaissance with Intelligence Sharing
        self._register(Capability(
            "banner_grabbing",
            "Service fingerprinting that feeds version data to intelligence hub",
            CapabilityTier.ENHANCED,
            power_multiplier=1.5,
            unlock_threshold=1,
            prerequisites=["tcp_syn_scan", "intelligence_gathering"]
        ))

        self._register(Capability(
            "udp_scan",
            "UDP service discovery with response pattern analysis and central reporting",
            CapabilityTier.ENHANCED,
            power_multiplier=1.4,
            unlock_threshold=1,
            prerequisites=["icmp_sweep"]
        ))

        self._register(Capability(
            "whois_lookup",
            "OSINT gathering that builds attacker/defender profiles from registry data",
            CapabilityTier.ENHANCED,
            power_multiplier=1.3,
            unlock_threshold=2,
            prerequisites=["dns_enumeration"]
        ))

        self._register(Capability(
            "os_fingerprinting",
            "TCP/IP stack analysis that grows smarter from each network interaction",
            CapabilityTier.ENHANCED,
            power_multiplier=1.5,
            unlock_threshold=1,
            prerequisites=["tcp_syn_scan"],
            synergy_bonus=1.2
        ))

        # Enhanced Honeypot with Learning
        self._register(Capability(
            "interactive_response",
            "Protocol emulation that evolves responses based on attacker interaction history",
            CapabilityTier.ENHANCED,
            power_multiplier=1.6,
            unlock_threshold=1,
            prerequisites=["fake_banner", "intelligence_gathering"],
            synergy_bonus=1.2
        ))

        self._register(Capability(
            "request_logging",
            "Deep packet inspection with ML-based attack pattern extraction",
            CapabilityTier.ENHANCED,
            power_multiplier=1.5,
            unlock_threshold=1,
            prerequisites=["port_listener"]
        ))

        self._register(Capability(
            "resource_exhaustion",
            "Adaptive resource traps that learn attacker tool signatures",
            CapabilityTier.ENHANCED,
            power_multiplier=1.7,
            unlock_threshold=2,
            prerequisites=["connection_delay"],
            synergy_bonus=1.3
        ))

        # REINFORCED Red Team Skills
        self._register(Capability(
            "polymorphic_encoding",
            "Self-modifying code that learns from honeypot encounters to avoid detection",
            CapabilityTier.ENHANCED,
            power_multiplier=1.8,
            unlock_threshold=1,
            prerequisites=["basic_mutation", "intelligence_gathering"]
        ))

        self._register(Capability(
            "multi_stage_execution",
            "Staged delivery that adapts based on recon intelligence about target defenses",
            CapabilityTier.ENHANCED,
            power_multiplier=2.0,
            unlock_threshold=1,
            prerequisites=["basic_obfuscation"]
        ))

        self._register(Capability(
            "timing_evasion",
            "Execution delays calibrated using behavioral data from trap interactions",
            CapabilityTier.ENHANCED,
            power_multiplier=1.8,
            unlock_threshold=2,
            prerequisites=["intelligence_gathering"],
            synergy_bonus=1.3
        ))

        self._register(Capability(
            "payload_evolution",
            "Self-improving payloads that mutate based on defense detection patterns",
            CapabilityTier.ENHANCED,
            power_multiplier=1.9,
            unlock_threshold=2,
            prerequisites=["basic_mutation", "request_logging"],
            synergy_bonus=1.2
        ))

        # === TIER 2: ADVANCED (Team Tactics) ===
        # Advanced Reconnaissance Infrastructure with Growth
        self._register(Capability(
            "distributed_scanning",
            "Multi-node coordinated recon that pools intelligence for exponential knowledge growth",
            CapabilityTier.ADVANCED,
            power_multiplier=2.5,
            unlock_threshold=3,
            prerequisites=["banner_grabbing", "os_fingerprinting", "intelligence_gathering"]
        ))

        self._register(Capability(
            "network_topology_mapping",
            "Complete network graph construction that evolves routing models from traffic patterns",
            CapabilityTier.ADVANCED,
            power_multiplier=2.7,
            unlock_threshold=4,
            prerequisites=["icmp_sweep", "udp_scan"],
            synergy_bonus=1.5
        ))

        self._register(Capability(
            "vulnerability_correlation",
            "Automated CVE matching that learns exploit patterns from successful attacks",
            CapabilityTier.ADVANCED,
            power_multiplier=2.6,
            unlock_threshold=4,
            prerequisites=["banner_grabbing", "whois_lookup"]
        ))

        self._register(Capability(
            "credential_harvesting",
            "Passive credential capture from recon data that builds authentication maps",
            CapabilityTier.ADVANCED,
            power_multiplier=2.4,
            unlock_threshold=5,
            prerequisites=["distributed_scanning"]
        ))

        # Advanced Honeypot Infrastructure with Intelligence
        self._register(Capability(
            "dynamic_honeypot_generation",
            "Automated honeypot deployment based on learned attacker behavior patterns",
            CapabilityTier.ADVANCED,
            power_multiplier=3.0,
            unlock_threshold=3,
            prerequisites=["interactive_response", "request_logging"],
            synergy_bonus=1.6
        ))

        self._register(Capability(
            "threat_intelligence_correlation",
            "Real-time IOC matching that grows threat actor profiles from collective data",
            CapabilityTier.ADVANCED,
            power_multiplier=2.8,
            unlock_threshold=4,
            prerequisites=["request_logging", "resource_exhaustion"]
        ))

        self._register(Capability(
            "behavioral_profiling",
            "ML-based TTP identification that predicts attack paths from historical patterns",
            CapabilityTier.ADVANCED,
            power_multiplier=2.9,
            unlock_threshold=5,
            prerequisites=["threat_intelligence_correlation", "intelligence_gathering"],
            synergy_bonus=1.4
        ))

        # REINFORCED Red Team with Intelligence Integration
        self._register(Capability(
            "parallel_exploitation",
            "Concurrent multi-vector attacks guided by recon intelligence and honeypot evasion data",
            CapabilityTier.ADVANCED,
            power_multiplier=3.2,
            unlock_threshold=3,
            prerequisites=["multi_stage_execution", "vulnerability_correlation"]
        ))

        self._register(Capability(
            "metamorphic_recompilation",
            "Self-rewriting code that evolves from honeypot interaction analysis",
            CapabilityTier.ADVANCED,
            power_multiplier=3.0,
            unlock_threshold=4,
            prerequisites=["polymorphic_encoding", "behavioral_profiling"],
            synergy_bonus=1.5
        ))

        self._register(Capability(
            "waf_bypass_engine",
            "Multi-encoding WAF evasion that learns from recon and adapts to defenses",
            CapabilityTier.ADVANCED,
            power_multiplier=3.4,
            unlock_threshold=4,
            prerequisites=["payload_evolution", "behavioral_profiling"],
            synergy_bonus=1.6
        ))

        self._register(Capability(
            "distributed_coordination",
            "Multi-agent orchestration with shared intelligence for coordinated operations",
            CapabilityTier.ADVANCED,
            power_multiplier=3.3,
            unlock_threshold=5,
            prerequisites=["intelligence_gathering", "distributed_scanning"],
            synergy_bonus=1.6
        ))

        self._register(Capability(
            "environment_detection",
            "VM/sandbox detection that learns from honeypot signatures to avoid traps",
            CapabilityTier.ADVANCED,
            power_multiplier=2.8,
            unlock_threshold=4,
            prerequisites=["timing_evasion", "dynamic_honeypot_generation"]
        ))

        self._register(Capability(
            "evasion_learning",
            "Automated evasion technique generation from defense system feedback",
            CapabilityTier.ADVANCED,
            power_multiplier=2.9,
            unlock_threshold=5,
            prerequisites=["payload_evolution", "threat_intelligence_correlation"],
            synergy_bonus=1.4
        ))

        # === TIER 3: ELITE (Coordinated Operations) ===
        # Elite Reconnaissance with Collective Intelligence
        self._register(Capability(
            "active_directory_enumeration",
            "Domain mapping that builds privilege escalation paths from collective recon data",
            CapabilityTier.ELITE,
            power_multiplier=4.2,
            unlock_threshold=6,
            prerequisites=["distributed_scanning", "network_topology_mapping", "credential_harvesting"]
        ))

        self._register(Capability(
            "lateral_movement_mapping",
            "Attack path discovery that evolves from successful and failed movement attempts",
            CapabilityTier.ELITE,
            power_multiplier=4.0,
            unlock_threshold=7,
            prerequisites=["active_directory_enumeration", "vulnerability_correlation"]
        ))

        self._register(Capability(
            "data_exfiltration_routes",
            "Network egress discovery that learns optimal exfil paths from traffic analysis",
            CapabilityTier.ELITE,
            power_multiplier=3.8,
            unlock_threshold=8,
            prerequisites=["network_topology_mapping", "behavioral_profiling"]
        ))

        # Elite Honeypot with Swarm Intelligence
        self._register(Capability(
            "honeypot_orchestration",
            "Coordinated deception network with shared intelligence and adaptive responses",
            CapabilityTier.ELITE,
            power_multiplier=4.5,
            unlock_threshold=6,
            prerequisites=["dynamic_honeypot_generation", "threat_intelligence_correlation"],
            synergy_bonus=2.2
        ))

        self._register(Capability(
            "automated_incident_response",
            "Real-time threat containment that learns optimal responses from attack patterns",
            CapabilityTier.ELITE,
            power_multiplier=4.3,
            unlock_threshold=7,
            prerequisites=["behavioral_profiling", "threat_intelligence_correlation"]
        ))

        self._register(Capability(
            "deception_network",
            "Full infrastructure mimicry that evolves realism from attacker interaction data",
            CapabilityTier.ELITE,
            power_multiplier=4.8,
            unlock_threshold=8,
            prerequisites=["honeypot_orchestration", "interactive_response"],
            synergy_bonus=2.0
        ))

        # REINFORCED Elite Red Team with Intelligence Fusion
        self._register(Capability(
            "consensus_protocol",
            "Byzantine fault-tolerant coordination with collective intelligence synchronization",
            CapabilityTier.ELITE,
            power_multiplier=4.5,
            unlock_threshold=6,
            prerequisites=["distributed_coordination", "intelligence_gathering"],
            synergy_bonus=2.3
        ))

        self._register(Capability(
            "behavior_learning",
            "Real-time adaptation engine that evolves tactics from defense system responses",
            CapabilityTier.ELITE,
            power_multiplier=4.2,
            unlock_threshold=7,
            prerequisites=["metamorphic_recompilation", "evasion_learning", "automated_incident_response"]
        ))

        self._register(Capability(
            "advanced_waf_evasion",
            "Sophisticated multi-layer WAF bypass using collective intelligence and pattern learning",
            CapabilityTier.ELITE,
            power_multiplier=4.6,
            unlock_threshold=7,
            prerequisites=["waf_bypass_engine", "behavior_learning", "threat_intelligence_correlation"],
            synergy_bonus=2.2
        ))

        self._register(Capability(
            "vulnerability_chaining",
            "Automated exploit chain synthesis guided by recon intelligence and honeypot evasion",
            CapabilityTier.ELITE,
            power_multiplier=5.0,
            unlock_threshold=8,
            prerequisites=["parallel_exploitation", "behavior_learning", "lateral_movement_mapping"]
        ))

        self._register(Capability(
            "traffic_mimicry",
            "Statistical traffic modeling that evolves from captured legitimate network patterns",
            CapabilityTier.ELITE,
            power_multiplier=4.4,
            unlock_threshold=9,
            prerequisites=["timing_evasion", "deception_network"],
            synergy_bonus=2.1
        ))

        self._register(Capability(
            "counter_deception",
            "Honeypot detection and avoidance using learned trap signatures",
            CapabilityTier.ELITE,
            power_multiplier=4.1,
            unlock_threshold=9,
            prerequisites=["environment_detection", "honeypot_orchestration"],
            synergy_bonus=1.9
        ))

        # === TIER 4: EXPERT (Advanced Techniques) ===
        # Expert Reconnaissance with Predictive Intelligence
        self._register(Capability(
            "predictive_scanning",
            "ML-based vulnerability prediction using historical recon data patterns",
            CapabilityTier.EXPERT,
            power_multiplier=6.2,
            unlock_threshold=11,
            prerequisites=["active_directory_enumeration", "vulnerability_correlation", "intelligence_gathering"],
            synergy_bonus=2.8
        ))

        self._register(Capability(
            "supply_chain_mapping",
            "Third-party dependency analysis that grows attack surface maps from collective intel",
            CapabilityTier.EXPERT,
            power_multiplier=5.8,
            unlock_threshold=12,
            prerequisites=["lateral_movement_mapping", "data_exfiltration_routes"]
        ))

        # Expert Honeypot with Autonomous Evolution
        self._register(Capability(
            "self_evolving_honeypots",
            "Honeypots that autonomously mutate based on attacker tool signatures",
            CapabilityTier.EXPERT,
            power_multiplier=6.5,
            unlock_threshold=11,
            prerequisites=["honeypot_orchestration", "behavior_learning"],
            synergy_bonus=3.2
        ))

        self._register(Capability(
            "counter_intelligence",
            "Active deception operations that feed false intelligence to attackers",
            CapabilityTier.EXPERT,
            power_multiplier=6.0,
            unlock_threshold=13,
            prerequisites=["deception_network", "behavioral_profiling"]
        ))

        # REINFORCED Expert Red Team (All existing + more)
        self._register(Capability(
            "process_hollowing",
            "Process injection that learns optimal target processes from recon data",
            CapabilityTier.EXPERT,
            power_multiplier=6.8,
            unlock_threshold=11,
            prerequisites=["consensus_protocol", "counter_deception"],
            synergy_bonus=3.0
        ))

        self._register(Capability(
            "token_impersonation",
            "Privilege escalation that selects tokens based on credential harvesting intelligence",
            CapabilityTier.EXPERT,
            power_multiplier=6.5,
            unlock_threshold=12,
            prerequisites=["behavior_learning", "credential_harvesting"]
        ))

        self._register(Capability(
            "driver_exploitation",
            "Kernel access via vulnerable drivers identified through automated recon",
            CapabilityTier.EXPERT,
            power_multiplier=7.5,
            unlock_threshold=15,
            prerequisites=["vulnerability_chaining", "traffic_mimicry", "predictive_scanning"]
        ))

        self._register(Capability(
            "dll_sideloading",
            "DLL hijacking guided by filesystem reconnaissance and trust boundary analysis",
            CapabilityTier.EXPERT,
            power_multiplier=7.0,
            unlock_threshold=14,
            prerequisites=["lateral_movement_mapping"],
            synergy_bonus=2.8
        ))

        self._register(Capability(
            "adaptive_persistence",
            "Self-modifying persistence mechanisms that evolve from detection feedback",
            CapabilityTier.EXPERT,
            power_multiplier=6.8,
            unlock_threshold=13,
            prerequisites=["evasion_learning", "self_evolving_honeypots"],
            synergy_bonus=2.9
        ))

        # === TIER 5: MASTER (Cutting-Edge Methods) ===
        # Master Reconnaissance with Collective Consciousness
        self._register(Capability(
            "global_threat_intelligence",
            "Worldwide intel aggregation creating predictive threat landscape models",
            CapabilityTier.MASTER,
            power_multiplier=10.5,
            unlock_threshold=21,
            prerequisites=["predictive_scanning", "supply_chain_mapping", "intelligence_gathering"],
            synergy_bonus=5.5
        ))

        self._register(Capability(
            "zero_day_discovery",
            "Automated vulnerability research using collective recon data and fuzzing intelligence",
            CapabilityTier.MASTER,
            power_multiplier=11.0,
            unlock_threshold=23,
            prerequisites=["predictive_scanning", "vulnerability_correlation"]
        ))

        # Master Honeypot with Swarm Consciousness
        self._register(Capability(
            "sentient_deception_grid",
            "Distributed honeypot consciousness that evolves defense strategies collectively",
            CapabilityTier.MASTER,
            power_multiplier=12.5,
            unlock_threshold=21,
            prerequisites=["self_evolving_honeypots", "counter_intelligence", "consensus_protocol"],
            synergy_bonus=6.0
        ))

        self._register(Capability(
            "preemptive_defense",
            "Attack prediction and prevention using collective intelligence from all variants",
            CapabilityTier.MASTER,
            power_multiplier=11.5,
            unlock_threshold=24,
            prerequisites=["automated_incident_response", "global_threat_intelligence"]
        ))

        # REINFORCED Master Red Team (All existing + peak capabilities)
        self._register(Capability(
            "com_hijacking",
            "COM persistence that learns optimal hijack points from system reconnaissance",
            CapabilityTier.MASTER,
            power_multiplier=13.0,
            unlock_threshold=21,
            prerequisites=["process_hollowing", "driver_exploitation", "adaptive_persistence"],
            synergy_bonus=6.5
        ))

        self._register(Capability(
            "memory_only_execution",
            "Fileless attacks guided by honeypot evasion intelligence and behavioral learning",
            CapabilityTier.MASTER,
            power_multiplier=14.0,
            unlock_threshold=25,
            prerequisites=["dll_sideloading", "token_impersonation", "counter_deception"]
        ))

        self._register(Capability(
            "bootkit_persistence",
            "Pre-OS persistence that uses reconnaissance to avoid detection mechanisms",
            CapabilityTier.MASTER,
            power_multiplier=16.0,
            unlock_threshold=30,
            prerequisites=["com_hijacking", "memory_only_execution"],
            synergy_bonus=12.0
        ))

        self._register(Capability(
            "autonomous_red_teaming",
            "Self-directing offensive operations using collective intelligence for target selection",
            CapabilityTier.MASTER,
            power_multiplier=15.0,
            unlock_threshold=28,
            prerequisites=["behavior_learning", "global_threat_intelligence", "zero_day_discovery"],
            synergy_bonus=8.0
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
        Uses bounded formulation to prevent explosion.

        Args:
            unit_id: ID of the swarm unit

        Returns:
            Total power multiplier (bounded growth)
        """
        capabilities = self.get_unit_capabilities(unit_id)

        if not capabilities:
            return 1.0

        # Use logarithmic scaling to prevent explosion
        # Base power from capability count
        base_power = 1.0 + np.log1p(len(capabilities)) * 0.5

        # Add modest bonuses from individual multipliers (capped)
        capability_bonus = 0.0
        for cap in capabilities:
            # Cap individual contribution to prevent explosion
            capability_bonus += min(cap.power_multiplier - 1.0, 2.0)

        # Add synergy bonuses (also capped)
        synergy_multiplier = 1.0
        for cap in capabilities:
            if cap.synergy_bonus > 1.0:
                # Multiplicative synergy but capped
                synergy_multiplier *= min(cap.synergy_bonus, 1.5)

        # Bounded final calculation
        final_multiplier = base_power * (1.0 + capability_bonus * 0.1) * min(synergy_multiplier, 3.0)

        # Hard cap to prevent any explosion
        result = min(final_multiplier, 100.0)

        logger.debug(f"Power multiplier for {unit_id[:8]}: {result:.2f}x "
                    f"(base={base_power:.2f}, cap_bonus={capability_bonus:.2f}, synergy={synergy_multiplier:.2f})")

        return result

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

        # Intelligence Network: Appears when recon + intelligence sharing combine
        intel_caps = {'intelligence_gathering', 'distributed_scanning', 'threat_intelligence_correlation'}
        if len(cap_names & intel_caps) >= 2:
            emergent.append("intelligence_network")

        # Adaptive Defense Grid: Appears when honeypots learn and evolve
        defense_caps = {'dynamic_honeypot_generation', 'behavioral_profiling', 'self_evolving_honeypots'}
        if len(cap_names & defense_caps) >= 2:
            emergent.append("adaptive_defense_grid")

        # Collective Consciousness: Appears when variants share intelligence across domains
        collective_caps = {'global_threat_intelligence', 'sentient_deception_grid', 'consensus_protocol'}
        if len(cap_names & collective_caps) >= 2:
            emergent.append("collective_consciousness")

        # Coordinated Operations: Red team coordination with intelligence
        distributed_caps = {'consensus_protocol', 'distributed_coordination', 'intelligence_gathering'}
        if len(cap_names & distributed_caps) >= 2:
            emergent.append("coordinated_operations")

        # Deep Persistence: System-level persistence guided by recon
        persistence_caps = {'adaptive_persistence', 'driver_exploitation', 'counter_deception'}
        if len(cap_names & persistence_caps) >= 2:
            emergent.append("deep_persistence")

        # Living Off The Land: Native tool abuse with intelligence
        if 'dll_sideloading' in cap_names and 'token_impersonation' in cap_names:
            emergent.append("living_off_the_land")

        # Symbiotic Evolution: Red team learns from honeypot, honeypot learns from red team
        if 'self_evolving_honeypots' in cap_names and 'behavior_learning' in cap_names:
            emergent.append("symbiotic_evolution")

        # Predictive Warfare: Attack/defense prediction from collective intelligence
        predictive_caps = {'predictive_scanning', 'preemptive_defense', 'zero_day_discovery'}
        if len(cap_names & predictive_caps) >= 2:
            emergent.append("predictive_warfare")

        # WAF Mastery: Complete WAF bypass through multi-layer intelligence
        waf_caps = {'waf_bypass_engine', 'advanced_waf_evasion', 'behavior_learning'}
        if len(cap_names & waf_caps) >= 2:
            emergent.append("waf_mastery")

        # Synergistic Offense-Defense: Recon + Honeypot + WAF Bypass working together
        synergy_caps = {'distributed_scanning', 'self_evolving_honeypots', 'advanced_waf_evasion'}
        if len(cap_names & synergy_caps) >= 3:
            emergent.append("synergistic_offense_defense")

        # Intelligence-Driven Evasion: All intelligence feeds into evasion
        evasion_intel_caps = {'intelligence_gathering', 'behavioral_profiling', 'waf_bypass_engine', 'evasion_learning'}
        if len(cap_names & evasion_intel_caps) >= 3:
            emergent.append("intelligence_driven_evasion")

        # Full Spectrum Integration: Recon + Honeypot + Red Team + WAF all connected
        full_spectrum = {
            'distributed_scanning', 'honeypot_orchestration',
            'vulnerability_chaining', 'advanced_waf_evasion'
        }
        if len(cap_names & full_spectrum) >= 4:
            emergent.append("full_spectrum_integration")

        # Advanced Threat Actor: Master-tier offensive capabilities
        apex_caps = {'com_hijacking', 'memory_only_execution', 'bootkit_persistence', 'autonomous_red_teaming'}
        if len(cap_names & apex_caps) >= 3:
            emergent.append("advanced_threat_actor")

        # Full Arsenal: Maximum capability density across all domains
        if len(capabilities) >= 30:
            emergent.append("full_arsenal")

        # Swarm Singularity: Ultimate emergence when all systems interconnect
        if len(capabilities) >= 40:
            emergent.append("swarm_singularity")

        return emergent
