"""
Variant Breeding System with Job-Based Lifecycles

Implements ephemeral variants that:
- Have one or more jobs based on merge tier (bigger merges = more jobs)
- Die after completing their jobs
- Send all learned intelligence back to central hub
- Breed cross-type variants (recon + honeypot + WAF, etc.)
- Produce progressively stronger offspring at higher tiers
- Specialize completely differently based on role
"""

import logging
from hive_zero_core.security import SecureRandom, InputValidator, AuditLogger, AccessController
from hive_zero_core.security.audit_logger import SecurityEvent
from hive_zero_core.security.access_control import OperationType

from typing import List, Dict, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import hashlib

from hive_zero_core.agents.genetic_operators import Individual
from hive_zero_core.agents.swarm_fusion import SwarmUnit, MergeStrategy
from hive_zero_core.agents.capability_escalation import CapabilityTier
from hive_zero_core.agents.stealth_backpack import (
    StealthBackpack, StealthLevel, CollectionMode
)


logger = logging.getLogger(__name__)


class VariantRole(Enum):
    """Specialized roles for variants with completely different capabilities."""
    RECONNAISSANCE = "reconnaissance"  # Network scanning, fingerprinting
    HONEYPOT = "honeypot"  # Defensive traps, deception
    WAF_BYPASS = "waf_bypass"  # Evasion techniques
    PAYLOAD_GEN = "payload_gen"  # Exploit generation
    STEALTH = "stealth"  # Anti-forensics, hiding
    EXFILTRATION = "exfiltration"  # Data extraction
    PERSISTENCE = "persistence"  # Long-term access
    LATERAL_MOVEMENT = "lateral_movement"  # Network traversal


class JobStatus(Enum):
    """Status of variant job execution."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class VariantJob:
    """Represents a single job/task for a variant."""
    job_id: str
    job_type: str  # e.g., "scan_subnet", "deploy_honeypot", "test_payload"
    target: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: JobStatus = JobStatus.PENDING
    intelligence_gathered: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    success: bool = False

    def __repr__(self):
        return f"Job({self.job_id[:8]}, type={self.job_type}, status={self.status.value})"


@dataclass
class Variant:
    """
    Ephemeral variant with job-based lifecycle.
    
    Lives only to complete assigned jobs, then dies and reports intelligence.
    Stronger parents (higher tier) produce stronger variants with more jobs.
    Equipped with StealthBackpack for infiltration/exfiltration operations.
    """
    variant_id: str
    role: VariantRole
    genome: str
    fitness: float
    generation: int
    tier: CapabilityTier
    parent_merge_count: int  # How many merges the parent had
    jobs: List[VariantJob] = field(default_factory=list)
    max_jobs: int = 1  # Determined by parent tier
    completed_jobs: int = 0
    intelligence_buffer: Dict[str, Any] = field(default_factory=dict)
    is_alive: bool = True
    specialization_traits: Dict[str, float] = field(default_factory=dict)
    cross_bred: bool = False  # True if bred from different role parents
    parent_roles: List[VariantRole] = field(default_factory=list)
    backpack: Optional[StealthBackpack] = None  # Stealth backpack for infil/exfil

    def __post_init__(self):
        if not self.variant_id:
            self.variant_id = SecureRandom.random_id(12)
        
        # Initialize role-specific specialization traits
        self._initialize_specialization()
        
        # Initialize stealth backpack based on role and tier
        self._initialize_backpack()

    def _initialize_specialization(self):
        """Initialize completely different traits based on role."""
        role_traits = {
            VariantRole.RECONNAISSANCE: {
                'scan_speed': 0.8,
                'stealth_level': 0.6,
                'coverage_breadth': 0.9,
                'fingerprint_accuracy': 0.7
            },
            VariantRole.HONEYPOT: {
                'deception_level': 0.8,
                'trap_sophistication': 0.7,
                'response_delay': 0.6,
                'intelligence_extraction': 0.9
            },
            VariantRole.WAF_BYPASS: {
                'evasion_creativity': 0.8,
                'encoding_depth': 0.7,
                'pattern_breaking': 0.9,
                'signature_mutation': 0.8
            },
            VariantRole.PAYLOAD_GEN: {
                'exploit_potency': 0.7,
                'obfuscation_level': 0.8,
                'polymorphism': 0.9,
                'stability': 0.6
            },
            VariantRole.STEALTH: {
                'footprint_minimization': 0.9,
                'log_evasion': 0.8,
                'memory_hiding': 0.7,
                'entropy_reduction': 0.8
            },
            VariantRole.EXFILTRATION: {
                'bandwidth_efficiency': 0.7,
                'covert_channel_usage': 0.8,
                'data_compression': 0.9,
                'protocol_mimicry': 0.8
            },
            VariantRole.PERSISTENCE: {
                'hiding_effectiveness': 0.8,
                'resilience': 0.9,
                'reinfection_capability': 0.7,
                'dormancy_control': 0.6
            },
            VariantRole.LATERAL_MOVEMENT: {
                'credential_harvesting': 0.8,
                'network_traversal': 0.9,
                'privilege_escalation': 0.7,
                'host_enumeration': 0.8
            }
        }
        
        self.specialization_traits = role_traits.get(self.role, {})
        
        # Scale traits by parent tier (higher tier = stronger offspring)
        tier_multiplier = 1.0 + (self.tier.value * 0.2)
        self.specialization_traits = {
            k: min(v * tier_multiplier, 1.0)
            for k, v in self.specialization_traits.items()
        }

    def _initialize_backpack(self):
        """Initialize stealth backpack based on role and tier."""
        # Determine stealth level based on tier
        stealth_mapping = {
            CapabilityTier.BASIC: StealthLevel.LOW,
            CapabilityTier.ENHANCED: StealthLevel.MEDIUM,
            CapabilityTier.ADVANCED: StealthLevel.HIGH,
            CapabilityTier.ELITE: StealthLevel.MAXIMUM,
            CapabilityTier.EXPERT: StealthLevel.MAXIMUM,
            CapabilityTier.MASTER: StealthLevel.MAXIMUM
        }
        
        # Determine collection mode based on role
        mode_mapping = {
            VariantRole.RECONNAISSANCE: CollectionMode.VACUUM,
            VariantRole.HONEYPOT: CollectionMode.PASSIVE,
            VariantRole.WAF_BYPASS: CollectionMode.MOSQUITO,
            VariantRole.PAYLOAD_GEN: CollectionMode.SURGICAL,
            VariantRole.STEALTH: CollectionMode.MOSQUITO,
            VariantRole.EXFILTRATION: CollectionMode.SURGICAL,
            VariantRole.PERSISTENCE: CollectionMode.PASSIVE,
            VariantRole.LATERAL_MOVEMENT: CollectionMode.VACUUM
        }
        
        stealth_level = stealth_mapping.get(self.tier, StealthLevel.MEDIUM)
        collection_mode = mode_mapping.get(self.role, CollectionMode.MOSQUITO)
        
        self.backpack = StealthBackpack(
            stealth_level=stealth_level,
            collection_mode=collection_mode
        )
        
        logger.debug(f"Variant {self.variant_id} backpack initialized: "
                    f"stealth={stealth_level.name}, mode={collection_mode.value}")

    def assign_job(self, job: VariantJob):
        """Assign a job to this variant."""
        if len(self.jobs) >= self.max_jobs:
            logger.warning(f"Variant {self.variant_id} already has max jobs ({self.max_jobs})")
            return False
        
        self.jobs.append(job)
        logger.info(f"Assigned job {job.job_id[:8]} to variant {self.variant_id}")
        return True

    def complete_job(self, job_id: str, intelligence: Dict[str, Any], success: bool = True):
        """
        Mark job as complete and store intelligence.
        
        Args:
            job_id: ID of the job to complete
            intelligence: Intelligence data gathered during job execution
            success: Whether the job succeeded
        """
        for job in self.jobs:
            if job.job_id == job_id:
                job.status = JobStatus.COMPLETED if success else JobStatus.FAILED
                job.success = success
                job.intelligence_gathered = intelligence
                
                if success:
                    self.completed_jobs += 1
                    # Store in intelligence buffer for central hub
                    self.intelligence_buffer[job_id] = {
                        'job_type': job.job_type,
                        'role': self.role.value,
                        'intelligence': intelligence,
                        'variant_id': self.variant_id,
                        'tier': self.tier.name,
                        'specialization': self.specialization_traits
                    }
                    logger.info(f"Variant {self.variant_id} completed job {job_id[:8]}")
                
                # Check if all jobs complete
                if self.all_jobs_complete():
                    self.die()
                
                return True
        
        logger.warning(f"Job {job_id} not found in variant {self.variant_id}")
        return False

    def all_jobs_complete(self) -> bool:
        """Check if all jobs are complete."""
        return all(job.status in [JobStatus.COMPLETED, JobStatus.FAILED] 
                  for job in self.jobs)

    def die(self):
        """Variant dies after completing all jobs. Harvests backpack intelligence."""
        self.is_alive = False
        
        # Harvest backpack intelligence before dying
        if self.backpack:
            backpack_intel = self.backpack.harvest_intelligence()
            self.intelligence_buffer['backpack_harvest'] = backpack_intel
            logger.debug(f"Variant {self.variant_id} harvested backpack intelligence: "
                        f"{backpack_intel['metrics']['total_collected']} items")
        
        logger.info(f"🪦 Variant {self.variant_id} (role={self.role.value}, tier={self.tier.name}) "
                   f"died after completing {self.completed_jobs}/{len(self.jobs)} jobs")

    def get_intelligence_report(self) -> Dict[str, Any]:
        """Generate final intelligence report to send to central hub."""
        report = {
            'variant_id': self.variant_id,
            'role': self.role.value,
            'tier': self.tier.name,
            'generation': self.generation,
            'parent_merge_count': self.parent_merge_count,
            'jobs_completed': self.completed_jobs,
            'total_jobs': len(self.jobs),
            'success_rate': self.completed_jobs / len(self.jobs) if self.jobs else 0.0,
            'intelligence': self.intelligence_buffer,
            'specialization_traits': self.specialization_traits,
            'cross_bred': self.cross_bred,
            'parent_roles': [r.value for r in self.parent_roles],
            'fitness': self.fitness
        }
        
        # Include backpack metrics if available
        if self.backpack:
            report['backpack_metrics'] = self.backpack.get_metrics().__dict__
        
        return report

    def __repr__(self):
        status = "💀" if not self.is_alive else "🔴" if self.jobs else "🟢"
        return (f"Variant{status}({self.variant_id[:8]}, role={self.role.value}, "
                f"tier={self.tier.name}, jobs={self.completed_jobs}/{len(self.jobs)})")


class IntelligenceHub:
    """
    Central hub for collecting intelligence from all variants.
    Aggregates learnings and provides collective intelligence.
    """

    def __init__(self):
        self.intelligence_store: Dict[str, List[Dict]] = {}
        self.role_statistics: Dict[VariantRole, Dict] = {}
        self.pattern_database: Dict[str, Any] = {}
        self.successful_techniques: Dict[str, int] = {}
        
    def receive_intelligence(self, variant_report: Dict[str, Any]):
        """
        Receive intelligence from a dead variant.
        
        Args:
            variant_report: Intelligence report from variant
        """
        role = variant_report['role']
        
        if role not in self.intelligence_store:
            self.intelligence_store[role] = []
        
        self.intelligence_store[role].append(variant_report)
        
        # Update statistics
        self._update_statistics(variant_report)
        
        # Extract patterns
        self._extract_patterns(variant_report)
        
        logger.info(f"📥 Received intelligence from variant {variant_report['variant_id'][:8]} "
                   f"(role={role}, success_rate={variant_report['success_rate']:.2f})")

    def _update_statistics(self, report: Dict[str, Any]):
        """Update role-specific statistics."""
        role = VariantRole(report['role'])
        
        if role not in self.role_statistics:
            self.role_statistics[role] = {
                'total_variants': 0,
                'total_jobs': 0,
                'successful_jobs': 0,
                'average_fitness': 0.0,
                'tier_distribution': {}
            }
        
        stats = self.role_statistics[role]
        stats['total_variants'] += 1
        stats['total_jobs'] += report['total_jobs']
        stats['successful_jobs'] += report['jobs_completed']
        
        # Running average of fitness
        n = stats['total_variants']
        stats['average_fitness'] = (stats['average_fitness'] * (n-1) + report['fitness']) / n
        
        # Tier distribution
        tier = report['tier']
        stats['tier_distribution'][tier] = stats['tier_distribution'].get(tier, 0) + 1

    def _extract_patterns(self, report: Dict[str, Any]):
        """Extract successful patterns from intelligence."""
        for job_id, job_data in report['intelligence'].items():
            if job_data.get('intelligence'):
                # Track successful techniques
                job_type = job_data['job_type']
                self.successful_techniques[job_type] = \
                    self.successful_techniques.get(job_type, 0) + 1

    def get_collective_intelligence(self, role: Optional[VariantRole] = None) -> Dict[str, Any]:
        """
        Get aggregated intelligence for a role or all roles.
        
        Args:
            role: Specific role to query, or None for all
            
        Returns:
            Collective intelligence data
        """
        if role:
            return {
                'role': role.value,
                'statistics': self.role_statistics.get(role, {}),
                'reports': self.intelligence_store.get(role.value, [])
            }
        
        return {
            'all_roles': {
                role.value: self.role_statistics.get(role, {})
                for role in VariantRole
            },
            'total_variants': sum(len(reports) for reports in self.intelligence_store.values()),
            'successful_techniques': self.successful_techniques,
            'pattern_database': self.pattern_database
        }


class VariantBreeder:
    """
    Breeds new variants with job-based lifecycles.
    
    Key features:
    - Higher tier parents produce stronger offspring
    - More merges = more jobs per variant
    - Cross-breeding different roles creates hybrid variants
    - Offspring inherit collective intelligence
    """

    def __init__(self, intelligence_hub: IntelligenceHub):
        self.intelligence_hub = intelligence_hub
        self.breeding_history: List[Dict] = []
        self.variant_registry: Dict[str, Variant] = {}

    def calculate_job_count(self, parent_merge_count: int, tier: CapabilityTier) -> int:
        """
        Calculate number of jobs based on parent's merge count and tier.
        More merges = more jobs.
        
        Args:
            parent_merge_count: Number of merges the parent has undergone
            tier: Capability tier of parent
            
        Returns:
            Number of jobs for offspring (1-10)
        """
        # Base jobs from tier
        tier_jobs = {
            CapabilityTier.BASIC: 1,
            CapabilityTier.ENHANCED: 2,
            CapabilityTier.ADVANCED: 3,
            CapabilityTier.ELITE: 5,
            CapabilityTier.EXPERT: 7,
            CapabilityTier.MASTER: 10
        }
        
        base_jobs = tier_jobs.get(tier, 1)
        
        # Additional jobs from merge count
        merge_bonus = parent_merge_count // 3  # +1 job per 3 merges
        
        total_jobs = min(base_jobs + merge_bonus, 10)  # Cap at 10 jobs
        
        return total_jobs

    def breed_variant(self, parent: SwarmUnit, role: VariantRole) -> Variant:
        """
        Breed a single-role variant from a parent unit.
        
        Args:
            parent: Parent swarm unit
            role: Role for the new variant
            
        Returns:
            New variant with job-based lifecycle
        """
        # Offspring is stronger based on parent tier
        offspring_fitness = parent.fitness * (1.0 + parent.merge_count * 0.1)
        
        # Determine tier (offspring inherits or slightly degrades)
        parent_tier = self._get_tier_for_merge_count(parent.merge_count)
        offspring_tier = parent_tier  # Could add logic for degradation/enhancement
        
        # Calculate job count
        max_jobs = self.calculate_job_count(parent.merge_count, offspring_tier)
        
        variant = Variant(
            variant_id="",
            role=role,
            genome=parent.genome,
            fitness=offspring_fitness,
            generation=parent.generation + 1,
            tier=offspring_tier,
            parent_merge_count=parent.merge_count,
            max_jobs=max_jobs,
            parent_roles=[role]
        )
        
        # Inject collective intelligence
        self._inject_intelligence(variant, role)
        
        self.variant_registry[variant.variant_id] = variant
        
        logger.info(f"🧬 Bred {variant} from parent (tier={parent_tier.name}, "
                   f"merges={parent.merge_count}) with {max_jobs} jobs")
        
        return variant

    def cross_breed_variants(self, parent1: SwarmUnit, parent2: SwarmUnit,
                            role1: VariantRole, role2: VariantRole) -> Variant:
        """
        Cross-breed two different role variants to create hybrid.
        
        Args:
            parent1: First parent unit
            parent2: Second parent unit  
            role1: Role of first parent
            role2: Role of second parent
            
        Returns:
            Hybrid variant with traits from both roles
        """
        # Average properties
        avg_fitness = (parent1.fitness + parent2.fitness) / 2.0
        avg_merge_count = (parent1.merge_count + parent2.merge_count) // 2
        
        # Higher tier between parents
        tier1 = self._get_tier_for_merge_count(parent1.merge_count)
        tier2 = self._get_tier_for_merge_count(parent2.merge_count)
        offspring_tier = tier1 if tier1.value > tier2.value else tier2
        
        # Hybrid gets more jobs (synergy bonus)
        max_jobs = self.calculate_job_count(avg_merge_count, offspring_tier)
        max_jobs = int(max_jobs * 1.5)  # 50% bonus for cross-breeding
        
        # Primary role is randomly selected
        primary_role = SecureRandom.random_choice([role1, role2])
        
        # Combine genomes
        merged_genome = self._merge_genomes(parent1.genome, parent2.genome)
        
        variant = Variant(
            variant_id="",
            role=primary_role,
            genome=merged_genome,
            fitness=avg_fitness * 1.2,  # Hybrid vigor
            generation=max(parent1.generation, parent2.generation) + 1,
            tier=offspring_tier,
            parent_merge_count=avg_merge_count,
            max_jobs=max_jobs,
            cross_bred=True,
            parent_roles=[role1, role2]
        )
        
        # Inject intelligence from both roles
        self._inject_intelligence(variant, role1)
        self._inject_intelligence(variant, role2)
        
        # Hybrid traits (blend specializations)
        self._blend_specializations(variant, role1, role2)
        
        self.variant_registry[variant.variant_id] = variant
        
        logger.info(f"🧬🧬 Cross-bred hybrid {variant} from {role1.value}+{role2.value} "
                   f"with {max_jobs} jobs (HYBRID VIGOR!)")
        
        return variant

    def _inject_intelligence(self, variant: Variant, role: VariantRole):
        """Inject collective intelligence from hub into variant."""
        collective_intel = self.intelligence_hub.get_collective_intelligence(role)
        
        # Boost variant traits based on collective learnings
        if 'statistics' in collective_intel:
            stats = collective_intel['statistics']
            if stats:
                # Variants learn from past successes
                success_boost = stats.get('successful_jobs', 0) / max(stats.get('total_jobs', 1), 1)
                variant.fitness *= (1.0 + success_boost * 0.2)
                
                logger.debug(f"Injected intelligence into {variant.variant_id[:8]}: "
                           f"success_boost={success_boost:.2f}")

    def _blend_specializations(self, variant: Variant, role1: VariantRole, role2: VariantRole):
        """Blend specialization traits from two different roles."""
        # This is a hybrid, so it gets traits from both roles
        role_traits_1 = variant.specialization_traits.copy()
        
        # Get traits for second role
        temp_variant = Variant(
            variant_id="temp",
            role=role2,
            genome="",
            fitness=0,
            generation=0,
            tier=variant.tier,
            parent_merge_count=0
        )
        role_traits_2 = temp_variant.specialization_traits
        
        # Blend: take average and add bonus
        blended = {}
        all_traits = set(role_traits_1.keys()) | set(role_traits_2.keys())
        
        for trait in all_traits:
            val1 = role_traits_1.get(trait, 0.5)
            val2 = role_traits_2.get(trait, 0.5)
            blended[trait] = min((val1 + val2) / 2.0 * 1.1, 1.0)  # 10% hybrid bonus
        
        variant.specialization_traits = blended
        
        logger.debug(f"Blended traits for hybrid: {len(blended)} traits")

    def _merge_genomes(self, genome1: str, genome2: str) -> str:
        """Merge two genomes for hybrid offspring."""
        # Simple interleaving strategy
        if len(genome1) < len(genome2):
            genome1, genome2 = genome2, genome1
        
        merged = []
        for i in range(len(genome2)):
            if random.random() < 0.5:
                merged.append(genome1[i])
            else:
                merged.append(genome2[i])
        
        # Append remaining from longer genome
        merged.extend(genome1[len(genome2):])
        
        return ''.join(merged)

    def _get_tier_for_merge_count(self, merge_count: int) -> CapabilityTier:
        """Determine tier based on merge count."""
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

    def spawn_generation(self, parents: List[SwarmUnit], 
                        roles: List[VariantRole],
                        cross_breed_rate: float = 0.3) -> List[Variant]:
        """
        Spawn a generation of variants from parent units.
        
        Args:
            parents: List of parent swarm units
            roles: List of roles to assign
            cross_breed_rate: Probability of cross-breeding (0.0-1.0)
            
        Returns:
            List of spawned variants
        """
        variants = []
        
        for parent in parents:
            # Decide whether to cross-breed
            if random.random() < cross_breed_rate and len(parents) > 1 and len(roles) > 1:
                # Cross-breed with another parent
                other_parent = SecureRandom.random_choice([p for p in parents if p != parent])
                role1 = SecureRandom.random_choice(roles)
                role2 = SecureRandom.random_choice([r for r in roles if r != role1])
                
                variant = self.cross_breed_variants(parent, other_parent, role1, role2)
            else:
                # Single-role breeding
                role = SecureRandom.random_choice(roles)
                variant = self.breed_variant(parent, role)
            
            variants.append(variant)
        
        logger.info(f"🧬 Spawned generation of {len(variants)} variants from {len(parents)} parents")
        
        return variants

    def harvest_intelligence(self, variant: Variant):
        """
        Harvest intelligence from a dead variant and send to central hub.
        
        Args:
            variant: Dead variant to harvest from
        """
        if variant.is_alive:
            logger.warning(f"Attempting to harvest from living variant {variant.variant_id}")
            return
        
        report = variant.get_intelligence_report()
        self.intelligence_hub.receive_intelligence(report)
        
        # Record breeding outcome
        self.breeding_history.append({
            'variant_id': variant.variant_id,
            'role': variant.role.value,
            'tier': variant.tier.name,
            'success_rate': report['success_rate'],
            'cross_bred': variant.cross_bred
        })
        
        logger.info(f"📤 Harvested intelligence from {variant}")

    def get_breeding_statistics(self) -> Dict[str, Any]:
        """Get statistics about breeding outcomes."""
        if not self.breeding_history:
            return {}
        
        total = len(self.breeding_history)
        cross_bred_count = sum(1 for h in self.breeding_history if h['cross_bred'])
        avg_success = sum(h['success_rate'] for h in self.breeding_history) / total
        
        tier_counts = {}
        for h in self.breeding_history:
            tier = h['tier']
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        return {
            'total_variants_bred': total,
            'cross_bred_variants': cross_bred_count,
            'cross_breed_rate': cross_bred_count / total,
            'average_success_rate': avg_success,
            'tier_distribution': tier_counts
        }
