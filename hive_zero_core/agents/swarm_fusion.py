"""
Swarm Fusion and Hierarchical Evolution

Implements agent merging, swarm intelligence, and hierarchical evolution
where individual units can combine into larger, more powerful collectives.
Enhanced with capability escalation - merged units gain exponentially more power.
"""

import logging
import random
from typing import List, Dict, Optional, Tuple, Set
import hashlib
from dataclasses import dataclass, field
from enum import Enum

from hive_zero_core.agents.genetic_operators import Individual
from hive_zero_core.agents.capability_escalation import CapabilityManager, EmergentBehaviors


logger = logging.getLogger(__name__)


class MergeStrategy(Enum):
    """Strategies for merging individuals."""
    CONCATENATE = "concatenate"
    INTERLEAVE = "interleave"
    BEST_SEGMENTS = "best_segments"
    NEURAL_BLEND = "neural_blend"
    HIERARCHICAL = "hierarchical"


@dataclass
class SwarmUnit:
    """
    Represents a collective unit that can contain multiple agents/individuals.
    Can be a single agent or a merged collective.
    Enhanced with capability tracking and power scaling.
    """
    id: str
    genome: str
    fitness: float
    generation: int
    members: List[str] = field(default_factory=list)  # IDs of constituent individuals
    merge_history: List[Dict] = field(default_factory=list)
    level: int = 0  # Hierarchy level (0=individual, 1=duo, 2=quad, etc.)
    specialization: Optional[str] = None
    merge_count: int = 0  # Total number of merges performed
    power_multiplier: float = 1.0  # Capability-based power scaling
    unlocked_capabilities: Set[str] = field(default_factory=set)
    emergent_behaviors: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.id:
            self.id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate unique ID for this unit."""
        content = f"{self.genome}{self.generation}{random.random()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def __repr__(self):
        member_count = len(self.members) if self.members else 1
        return (f"SwarmUnit(id={self.id[:8]}, level={self.level}, "
                f"members={member_count}, fitness={self.fitness:.3f}, "
                f"merges={self.merge_count}, power={self.power_multiplier:.1f}x)")


class SwarmFusion:
    """
    Handles merging of individuals and swarm units into larger collectives.
    Implements various fusion strategies and maintains swarm hierarchy.
    Enhanced with capability escalation - merged units gain exponentially more power.
    """

    def __init__(self, min_fitness_threshold: float = 0.5,
                 max_unit_size: int = 8):
        """
        Initialize swarm fusion engine.

        Args:
            min_fitness_threshold: Minimum fitness required for merging
            max_unit_size: Maximum number of members in a merged unit
        """
        self.min_fitness_threshold = min_fitness_threshold
        self.max_unit_size = max_unit_size
        self.swarm_registry: Dict[str, SwarmUnit] = {}
        self.merge_count = 0

        # Capability management
        self.capability_manager = CapabilityManager()

    def _apply_capability_escalation(self, unit: SwarmUnit) -> SwarmUnit:
        """
        Apply capability escalation to a unit based on merge count.
        THE MORE MERGES, THE MORE POWERFUL IT BECOMES.
        
        Args:
            unit: SwarmUnit to enhance
            
        Returns:
            Enhanced unit with new capabilities
        """
        # Unlock capabilities based on merge count
        newly_unlocked = self.capability_manager.unlock_capabilities_for_unit(
            unit.id,
            unit.merge_count,
            unit.unlocked_capabilities
        )
        
        if newly_unlocked:
            logger.info(f"⚡ Unit {unit.id[:8]} gained {len(newly_unlocked)} new capabilities!")
            for cap in newly_unlocked:
                logger.info(f"   └─ {cap.name} ({cap.tier.name}): {cap.description}")
        
        # Update unit's capabilities
        unit.unlocked_capabilities = self.capability_manager.unit_capabilities.get(unit.id, set())
        
        # Calculate new power multiplier
        unit.power_multiplier = self.capability_manager.calculate_power_multiplier(unit.id)
        
        # Check for emergent behaviors
        all_capabilities = self.capability_manager.get_unit_capabilities(unit.id)
        unit.emergent_behaviors = EmergentBehaviors.check_emergent_behavior(all_capabilities)
        
        if unit.emergent_behaviors:
            logger.info(f"🌟 EMERGENT BEHAVIOR DETECTED in {unit.id[:8]}: {', '.join(unit.emergent_behaviors)}")

        # Don't modify fitness with power_multiplier to avoid fitness inflation
        # Keep power as separate dimension for selection/merging heuristics

        return unit

    def merge_individuals(self, individual1: Individual, individual2: Individual,
                         strategy: MergeStrategy = MergeStrategy.BEST_SEGMENTS,
                         generation: int = 0) -> SwarmUnit:
        """
        Merge two individuals into a swarm unit.

        Args:
            individual1: First individual
            individual2: Second individual
            strategy: Merging strategy to use
            generation: Current generation number

        Returns:
            New SwarmUnit combining both individuals
        """
        # Check fitness threshold
        avg_fitness = (individual1.fitness + individual2.fitness) / 2
        if avg_fitness < self.min_fitness_threshold:
            logger.warning(f"Low fitness pair ({avg_fitness:.3f}), merge may not be beneficial")

        # Apply merge strategy
        if strategy == MergeStrategy.CONCATENATE:
            merged_genome = self._merge_concatenate(individual1.genome, individual2.genome)
        elif strategy == MergeStrategy.INTERLEAVE:
            merged_genome = self._merge_interleave(individual1.genome, individual2.genome)
        elif strategy == MergeStrategy.BEST_SEGMENTS:
            merged_genome = self._merge_best_segments(
                individual1.genome, individual2.genome,
                individual1.fitness, individual2.fitness
            )
        elif strategy == MergeStrategy.HIERARCHICAL:
            merged_genome = self._merge_hierarchical(individual1.genome, individual2.genome)
        else:
            # Default: concatenate
            merged_genome = self._merge_concatenate(individual1.genome, individual2.genome)

        # Calculate merged fitness (optimistic: better than average)
        merged_fitness = max(individual1.fitness, individual2.fitness) * 1.1

        # Create swarm unit
        unit = SwarmUnit(
            id="",
            genome=merged_genome,
            fitness=merged_fitness,
            generation=generation,
            members=[
                str(getattr(individual1, 'gene_seed', f"ind_{hash(individual1.genome)%10000}")),
                str(getattr(individual2, 'gene_seed', f"ind_{hash(individual2.genome)%10000}"))
            ],
            level=1,
            merge_count=1,  # First merge
            merge_history=[{
                'generation': generation,
                'strategy': strategy.value,
                'parent_fitnesses': [individual1.fitness, individual2.fitness]
            }]
        )

        # APPLY CAPABILITY ESCALATION
        unit = self._apply_capability_escalation(unit)

        self.swarm_registry[unit.id] = unit
        self.merge_count += 1

        logger.info(f"Merged individuals into {unit} using {strategy.value}")

        return unit

    def merge_units(self, unit1: SwarmUnit, unit2: SwarmUnit,
                   strategy: MergeStrategy = MergeStrategy.HIERARCHICAL,
                   generation: int = 0) -> SwarmUnit:
        """
        Merge two swarm units into a larger collective.

        Args:
            unit1: First swarm unit
            unit2: Second swarm unit
            strategy: Merging strategy
            generation: Current generation number

        Returns:
            Larger merged SwarmUnit
        """
        # Check size constraints
        if len(unit1.members) + len(unit2.members) > self.max_unit_size:
            logger.warning(f"Merge would exceed max_unit_size")

        # Hierarchical merge - preserve structure
        if strategy == MergeStrategy.HIERARCHICAL:
            merged_genome = self._merge_hierarchical(unit1.genome, unit2.genome)
        else:
            merged_genome = self._merge_best_segments(
                unit1.genome, unit2.genome,
                unit1.fitness, unit2.fitness
            )

        # Synergistic fitness (units work together better)
        synergy_bonus = 1.15 if unit1.level == unit2.level else 1.1
        merged_fitness = max(unit1.fitness, unit2.fitness) * synergy_bonus

        # Calculate total merge count (CUMULATIVE POWER)
        total_merge_count = unit1.merge_count + unit2.merge_count + 1

        # Create higher-level unit
        new_unit = SwarmUnit(
            id="",
            genome=merged_genome,
            fitness=merged_fitness,
            generation=generation,
            members=(unit1.members + unit2.members)[:self.max_unit_size],
            level=max(unit1.level, unit2.level) + 1,
            merge_count=total_merge_count,  # CUMULATIVE MERGE COUNT
            unlocked_capabilities=unit1.unlocked_capabilities | unit2.unlocked_capabilities,  # Inherit all capabilities
            merge_history=unit1.merge_history + unit2.merge_history + [{
                'generation': generation,
                'strategy': strategy.value,
                'merged_units': [unit1.id, unit2.id],
                'synergy_bonus': synergy_bonus,
                'total_merge_count': total_merge_count
            }]
        )

        # APPLY CAPABILITY ESCALATION (THE MORE MERGES, THE MORE POWER)
        new_unit = self._apply_capability_escalation(new_unit)

        self.swarm_registry[new_unit.id] = new_unit
        self.merge_count += 1

        logger.info(f"Merged units into {new_unit} (level {new_unit.level})")

        return new_unit

    def _merge_concatenate(self, genome1: str, genome2: str) -> str:
        """Concatenate genomes with separator."""
        separator = "\n# === MERGE BOUNDARY ===\n"
        return genome1 + separator + genome2

    def _merge_interleave(self, genome1: str, genome2: str) -> str:
        """Interleave lines from both genomes."""
        lines1 = genome1.split('\n')
        lines2 = genome2.split('\n')

        merged = []
        max_len = max(len(lines1), len(lines2))

        for i in range(max_len):
            if i < len(lines1):
                merged.append(lines1[i])
            if i < len(lines2):
                merged.append(lines2[i])

        return '\n'.join(merged)

    def _merge_best_segments(self, genome1: str, genome2: str,
                            fitness1: float, fitness2: float) -> str:
        """Merge by selecting best segments based on fitness."""
        # Weight ratio
        total_fitness = fitness1 + fitness2
        if total_fitness == 0:
            ratio = 0.5
        else:
            ratio = fitness1 / total_fitness

        lines1 = genome1.split('\n')
        lines2 = genome2.split('\n')

        merged = []
        max_len = max(len(lines1), len(lines2))

        for i in range(max_len):
            # Probabilistic selection based on fitness
            if random.random() < ratio:
                if i < len(lines1):
                    merged.append(lines1[i])
                elif i < len(lines2):
                    merged.append(lines2[i])
            else:
                if i < len(lines2):
                    merged.append(lines2[i])
                elif i < len(lines1):
                    merged.append(lines1[i])

        return '\n'.join(merged)

    def _merge_hierarchical(self, genome1: str, genome2: str) -> str:
        """
        Hierarchical merge - create wrapper that delegates to sub-genomes.
        Useful for code that creates a coordinated system.
        """
        wrapper = f'''# === HIERARCHICAL SWARM UNIT ===
# Collective intelligence combining multiple evolved strategies

def _strategy_alpha():
    """First merged strategy."""
{self._indent_code(genome1, 4)}

def _strategy_beta():
    """Second merged strategy."""
{self._indent_code(genome2, 4)}

def execute_swarm():
    """Coordinate execution of all strategies."""
    results = []
    try:
        results.append(_strategy_alpha())
    except Exception:
        pass
    try:
        results.append(_strategy_beta())
    except Exception:
        pass
    return results

# Execute only when run as a script
if __name__ == "__main__":
    execute_swarm()
'''
        return wrapper

    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent code by specified number of spaces."""
        indent = ' ' * spaces
        return '\n'.join(indent + line if line.strip() else line
                        for line in code.split('\n'))

    def create_mega_unit(self, units: List[SwarmUnit],
                        generation: int = 0) -> SwarmUnit:
        """
        Create a mega-unit from multiple swarm units.
        This is the highest level of hierarchy.

        Args:
            units: List of swarm units to merge
            generation: Current generation

        Returns:
            Mega SwarmUnit
        """
        if len(units) < 2:
            raise ValueError("Need at least 2 units to create mega-unit")

        logger.info(f"Creating mega-unit from {len(units)} swarm units")

        # Hierarchically merge all units
        current = units[0]
        for i in range(1, len(units)):
            current = self.merge_units(
                current, units[i],
                strategy=MergeStrategy.HIERARCHICAL,
                generation=generation
            )

        # Mark as mega-unit
        current.specialization = "mega_unit"

        logger.info(f"Mega-unit created: {current}")

        return current

    def get_lineage(self, unit_id: str) -> List[str]:
        """
        Get the full lineage of a swarm unit.

        Args:
            unit_id: ID of the unit

        Returns:
            List of ancestor IDs
        """
        if unit_id not in self.swarm_registry:
            return []

        unit = self.swarm_registry[unit_id]
        lineage = [unit_id]

        # Trace back through merge history
        for merge_event in unit.merge_history:
            if 'merged_units' in merge_event:
                for ancestor_id in merge_event['merged_units']:
                    lineage.extend(self.get_lineage(ancestor_id))

        return lineage

    def get_statistics(self) -> Dict:
        """Get statistics about swarm fusion with capability escalation data."""
        if not self.swarm_registry:
            return {'total_units': 0, 'merge_count': 0}

        units = list(self.swarm_registry.values())

        # Calculate power statistics
        power_multipliers = [u.power_multiplier for u in units]
        max_merge_count = max((u.merge_count for u in units), default=0)
        
        # Get capability statistics
        capability_stats = self.capability_manager.get_statistics()

        return {
            'total_units': len(units),
            'merge_count': self.merge_count,
            'max_level': max((u.level for u in units), default=0),
            'max_merge_count': max_merge_count,
            'avg_fitness': sum(u.fitness for u in units) / len(units),
            'mega_units': sum(1 for u in units if u.specialization == "mega_unit"),
            'level_distribution': self._count_by_level(units),
            'power_scaling': {
                'min_power': min(power_multipliers, default=1.0),
                'max_power': max(power_multipliers, default=1.0),
                'avg_power': sum(power_multipliers) / len(power_multipliers) if power_multipliers else 1.0,
            },
            'emergent_behaviors': sum(len(u.emergent_behaviors) for u in units),
            'capability_unlocks': capability_stats,
        }

    def _count_by_level(self, units: List[SwarmUnit]) -> Dict[int, int]:
        """Count units by hierarchy level."""
        counts = {}
        for unit in units:
            counts[unit.level] = counts.get(unit.level, 0) + 1
        return counts


class CollectiveIntelligence:
    """
    Manages collective intelligence across swarm units.
    Implements knowledge sharing, specialization, and coordination.
    """

    def __init__(self):
        """Initialize collective intelligence manager."""
        self.knowledge_base: Dict[str, Set[str]] = {}  # unit_id -> set of learned patterns
        self.specializations: Dict[str, List[str]] = {}  # specialization -> list of unit_ids
        self.coordination_matrix: Dict[Tuple[str, str], float] = {}  # (unit1, unit2) -> synergy score

    def share_knowledge(self, donor_unit: SwarmUnit, recipient_unit: SwarmUnit) -> bool:
        """
        Share knowledge from one unit to another.

        Args:
            donor_unit: Unit sharing knowledge
            recipient_unit: Unit receiving knowledge

        Returns:
            True if knowledge was successfully shared
        """
        # Extract "knowledge" (unique patterns in genome)
        donor_patterns = set(self._extract_patterns(donor_unit.genome))
        recipient_patterns = set(self._extract_patterns(recipient_unit.genome))

        # Find novel patterns
        novel_patterns = donor_patterns - recipient_patterns

        if novel_patterns:
            # Update knowledge base
            if recipient_unit.id not in self.knowledge_base:
                self.knowledge_base[recipient_unit.id] = set()

            self.knowledge_base[recipient_unit.id].update(novel_patterns)

            logger.info(f"Shared {len(novel_patterns)} patterns: {donor_unit.id[:8]} → {recipient_unit.id[:8]}")
            return True

        return False

    def _extract_patterns(self, genome: str, min_length: int = 5) -> List[str]:
        """Extract significant patterns from genome."""
        # Simple: split by lines and filter
        patterns = []
        for line in genome.split('\n'):
            line = line.strip()
            if len(line) >= min_length and not line.startswith('#'):
                patterns.append(line)

        return patterns

    def assign_specialization(self, unit: SwarmUnit, specialization: str):
        """
        Assign a specialization to a swarm unit.

        Args:
            unit: Swarm unit
            specialization: Type of specialization (e.g., 'evasion', 'obfuscation', 'stealth')
        """
        unit.specialization = specialization

        if specialization not in self.specializations:
            self.specializations[specialization] = []

        if unit.id not in self.specializations[specialization]:
            self.specializations[specialization].append(unit.id)

        logger.info(f"Unit {unit.id[:8]} specialized in: {specialization}")

    def calculate_synergy(self, unit1: SwarmUnit, unit2: SwarmUnit) -> float:
        """
        Calculate synergy score between two units.
        Higher score means better collaboration potential.

        Args:
            unit1: First unit
            unit2: Second unit

        Returns:
            Synergy score (0.0 to 1.0)
        """
        key = (unit1.id, unit2.id)

        # Check cache
        if key in self.coordination_matrix:
            return self.coordination_matrix[key]

        # Calculate synergy based on:
        # 1. Complementary specializations
        # 2. Shared knowledge
        # 3. Compatible fitness levels

        score = 0.0

        # Specialization complementarity
        if unit1.specialization and unit2.specialization:
            if unit1.specialization != unit2.specialization:
                score += 0.4  # Different specializations = complementary
            else:
                score += 0.2  # Same specialization = less synergy

        # Knowledge sharing potential
        patterns1 = set(self._extract_patterns(unit1.genome))
        patterns2 = set(self._extract_patterns(unit2.genome))

        overlap = len(patterns1 & patterns2)
        unique = len(patterns1 | patterns2)

        if unique > 0:
            diversity_score = 1 - (overlap / unique)
            score += diversity_score * 0.4

        # Fitness compatibility (similar fitness = better teamwork)
        fitness_diff = abs(unit1.fitness - unit2.fitness)
        fitness_score = max(0, 1 - fitness_diff)
        score += fitness_score * 0.2

        # Cache result
        self.coordination_matrix[key] = score

        logger.debug(f"Synergy {unit1.id[:8]} ↔ {unit2.id[:8]}: {score:.3f}")

        return score

    def form_optimal_team(self, units: List[SwarmUnit], team_size: int = 4) -> List[SwarmUnit]:
        """
        Form an optimal team from available units based on synergy.

        Args:
            units: Pool of available units
            team_size: Desired team size

        Returns:
            Optimized team of units
        """
        if len(units) <= team_size:
            return units

        # Greedy algorithm: start with highest fitness, add complementary units
        sorted_units = sorted(units, key=lambda u: u.fitness, reverse=True)

        team = [sorted_units[0]]

        for _ in range(team_size - 1):
            best_candidate = None
            best_avg_synergy = -1

            for candidate in sorted_units:
                if candidate in team:
                    continue

                # Calculate average synergy with current team
                synergies = [self.calculate_synergy(candidate, member) for member in team]
                avg_synergy = sum(synergies) / len(synergies)

                if avg_synergy > best_avg_synergy:
                    best_avg_synergy = avg_synergy
                    best_candidate = candidate

            if best_candidate:
                team.append(best_candidate)

        logger.info(f"Formed optimal team of {len(team)} units (avg synergy: {best_avg_synergy:.3f})")

        return team

    def get_collective_stats(self) -> Dict:
        """Get statistics about collective intelligence."""
        return {
            'total_units_with_knowledge': len(self.knowledge_base),
            'total_patterns_learned': sum(len(patterns) for patterns in self.knowledge_base.values()),
            'specializations': {k: len(v) for k, v in self.specializations.items()},
            'coordination_matrix_size': len(self.coordination_matrix),
        }
