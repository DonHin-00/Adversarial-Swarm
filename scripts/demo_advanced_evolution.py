#!/usr/bin/env python3
"""
Advanced Genetic Evolution Demo: Population-Based and Swarm Fusion

Demonstrates population evolution, swarm fusion, and collective intelligence capabilities.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def demo_population_evolution():
    """Demonstrate population-based genetic algorithm."""
    print("=" * 70)
    print("DEMO: Population-Based Genetic Evolution")
    print("=" * 70)

    from hive_zero_core.agents.population_evolution import PopulationManager
    from hive_zero_core.agents.genetic_evolution import NaturalSelection

    # Create population manager
    manager = PopulationManager(
        population_size=10,
        elite_size=2,
        mutation_rate=0.4,
        crossover_rate=0.7,
        max_generations=15
    )

    # Seed genome
    seed = """def exploit():
    payload = "test"
    return payload"""

    validator = lambda code: NaturalSelection.validate_python(code, strict=False)

    print(f"\n[1] Seed Genome:")
    print("-" * 70)
    print(seed)

    print(f"\n[2] Evolving population for 15 generations...")
    best = manager.evolve(seed, validator, generations=15)

    print(f"\n[3] Best Individual:")
    print("-" * 70)
    print(f"Fitness: {best.fitness:.3f}")
    print(f"Generation: {best.generation}")
    print(f"Genome (first 200 chars):")
    print(best.genome[:200] + ("..." if len(best.genome) > 200 else ""))

    print(f"\n[4] Evolution Statistics:")
    print("-" * 70)
    stats = manager.get_statistics()
    print(f"  Total generations: {stats['total_generations']}")
    print(f"  Final best fitness: {stats['final_best_fitness']:.3f}")
    print(f"  Fitness improvement: {stats['fitness_improvement']:.3f}")
    print(f"  Final diversity: {stats['final_diversity']} unique genomes")

    if stats.get('convergence_generation'):
        print(f"  Converged at generation: {stats['convergence_generation']}")


def demo_swarm_fusion():
    """Demonstrate swarm fusion and merging."""
    print("\n" + "=" * 70)
    print("DEMO: Swarm Fusion - Merging Units")
    print("=" * 70)

    from hive_zero_core.agents.swarm_fusion import SwarmFusion, MergeStrategy
    from hive_zero_core.agents.genetic_operators import Individual

    # Create fusion engine
    fusion = SwarmFusion(min_fitness_threshold=0.3, max_unit_size=8)

    # Create some individuals
    individuals = [
        Individual("def attack1(): return 'payload_a'", 0.7, 0),
        Individual("def attack2(): return 'payload_b'", 0.8, 0),
        Individual("def attack3(): return 'payload_c'", 0.75, 0),
        Individual("def attack4(): return 'payload_d'", 0.6, 0),
    ]

    print("\n[1] Initial Individuals:")
    print("-" * 70)
    for i, ind in enumerate(individuals):
        print(f"  Individual {i+1}: fitness={ind.fitness:.2f}")

    print("\n[2] Merging pairs using different strategies...")
    print("-" * 70)

    # Merge first pair with BEST_SEGMENTS
    unit1 = fusion.merge_individuals(
        individuals[0], individuals[1],
        strategy=MergeStrategy.BEST_SEGMENTS,
        generation=1
    )
    print(f"  Unit 1 (BEST_SEGMENTS): {unit1}")

    # Merge second pair with HIERARCHICAL
    unit2 = fusion.merge_individuals(
        individuals[2], individuals[3],
        strategy=MergeStrategy.HIERARCHICAL,
        generation=1
    )
    print(f"  Unit 2 (HIERARCHICAL): {unit2}")

    print("\n[3] Merging units into bigger collective...")
    print("-" * 70)

    mega_unit = fusion.merge_units(unit1, unit2, generation=2)
    print(f"  Mega Unit: {mega_unit}")
    print(f"  Total members: {len(mega_unit.members)}")
    print(f"  Hierarchy level: {mega_unit.level}")

    print("\n[4] Fusion Statistics:")
    print("-" * 70)
    stats = fusion.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")


def demo_collective_intelligence():
    """Demonstrate collective intelligence and team formation."""
    print("\n" + "=" * 70)
    print("DEMO: Collective Intelligence")
    print("=" * 70)

    from hive_zero_core.agents.swarm_fusion import CollectiveIntelligence, SwarmUnit

    # Create collective intelligence manager
    collective = CollectiveIntelligence()

    # Create some swarm units
    units = [
        SwarmUnit("", "code_a = 1\nattack()", 0.8, 0, level=1),
        SwarmUnit("", "code_b = 2\nevade()", 0.7, 0, level=1),
        SwarmUnit("", "code_c = 3\nsteal()", 0.9, 0, level=1),
        SwarmUnit("", "code_d = 4\nhide()", 0.6, 0, level=1),
    ]

    print("\n[1] Assigning Specializations:")
    print("-" * 70)

    specializations = ['evasion', 'obfuscation', 'stealth', 'persistence']
    for unit, spec in zip(units, specializations):
        collective.assign_specialization(unit, spec)
        print(f"  {unit.id[:8]}: {spec}")

    print("\n[2] Calculating Synergies:")
    print("-" * 70)

    for i in range(len(units)):
        for j in range(i + 1, len(units)):
            synergy = collective.calculate_synergy(units[i], units[j])
            print(f"  {units[i].id[:8]} ↔ {units[j].id[:8]}: {synergy:.3f}")

    print("\n[3] Forming Optimal Team:")
    print("-" * 70)

    team = collective.form_optimal_team(units, team_size=3)
    print(f"  Team size: {len(team)}")
    for member in team:
        print(f"    • {member.id[:8]} (fitness={member.fitness:.2f}, spec={member.specialization})")

    print("\n[4] Knowledge Sharing:")
    print("-" * 70)

    shared = collective.share_knowledge(units[2], units[3])  # High fitness shares with low
    print(f"  Knowledge shared: {shared}")

    stats = collective.get_collective_stats()
    print(f"  Total patterns learned: {stats['total_patterns_learned']}")
    print(f"  Specializations: {stats['specializations']}")


def demo_mega_unit_creation():
    """Demonstrate creating mega-units from multiple swarm units."""
    print("\n" + "=" * 70)
    print("DEMO: Mega-Unit Creation")
    print("=" * 70)

    from hive_zero_core.agents.swarm_fusion import SwarmFusion, SwarmUnit

    fusion = SwarmFusion()

    # Create several swarm units
    units = []
    for i in range(5):
        unit = SwarmUnit(
            "",
            f"# Strategy {i+1}\ndef execute():\n    pass",
            0.6 + i * 0.05,
            0,
            members=[f"agent_{i}"],
            level=1
        )
        units.append(unit)
        fusion.swarm_registry[unit.id] = unit

    print(f"\n[1] Created {len(units)} individual swarm units")
    print("-" * 70)

    for i, unit in enumerate(units):
        print(f"  Unit {i+1}: {unit}")

    print(f"\n[2] Creating Mega-Unit...")
    print("-" * 70)

    mega = fusion.create_mega_unit(units, generation=5)
    print(f"  {mega}")
    print(f"  Total members: {len(mega.members)}")
    print(f"  Hierarchy level: {mega.level}")
    print(f"  Specialization: {mega.specialization}")

    print(f"\n[3] Mega-Unit Genome Structure:")
    print("-" * 70)

    lines = mega.genome.split('\n')
    print(f"  Total lines: {len(lines)}")
    print(f"  Contains 'HIERARCHICAL SWARM UNIT': {'HIERARCHICAL SWARM UNIT' in mega.genome}")
    print(f"  Contains 'execute_swarm': {'execute_swarm' in mega.genome}")

    print(f"\n[4] Lineage Tracing:")
    print("-" * 70)

    lineage = fusion.get_lineage(mega.id)
    print(f"  Total ancestors: {len(lineage)}")
    print(f"  Lineage depth: {len(set(lineage))}")


def main():
    """Run all advanced demonstrations."""
    print("\n" + "#" * 70)
    print("# Advanced Genetic Evolution Demo")
    print("# Population-Based, Swarm Fusion & Collective Intelligence")
    print("#" * 70)

    try:
        demo_population_evolution()
        demo_swarm_fusion()
        demo_collective_intelligence()
        demo_mega_unit_creation()

        print("\n" + "=" * 70)
        print("All advanced demonstrations completed successfully!")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
