#!/usr/bin/env python3
"""
Demonstration script for genetic evolution capabilities in Adversarial-Swarm.

This showcases the polymorphic engine, natural selection, and generation tracking
for red team payload evolution.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hive_zero_core.agents.genetic_evolution import GeneticEvolution


def demo_code_evolution():
    """Demonstrate code polymorphism through genetic evolution."""
    print("=" * 70)
    print("DEMO: Code Polymorphism via Genetic Evolution")
    print("=" * 70)

    evolution = GeneticEvolution()

    # Original attack payload code
    original_code = """
def exploit_xss(target_url):
    import requests
    payload = "<script>alert('XSS')</script>"
    response = requests.post(target_url, data={'input': payload})
    return response.status_code == 200
"""

    print("\n[1] Original Code:")
    print("-" * 70)
    print(original_code)

    # Evolve the code
    print("\n[2] Evolving code through 3 generations...")
    generations = []
    for i in range(3):
        mutated, gene_seed, success = evolution.evolve_code(original_code, max_attempts=10)
        if success:
            generations.append((i + 1, gene_seed, mutated))
            print(f"  ✓ Generation {i + 1}: gene_seed={gene_seed}")

    # Show first mutated generation
    if generations:
        print(f"\n[3] Mutated Code (Generation 1, gene_seed={generations[0][1]}):")
        print("-" * 70)
        print(generations[0][2])

    # Show evolution stats
    print("\n[4] Evolution Statistics:")
    print("-" * 70)
    stats = evolution.get_stats()
    print(f"  Total mutations: {stats['total']}")
    print(f"  Successful: {stats['successful']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    print(f"  Current generation: {stats['current_generation']}")


def demo_payload_evolution():
    """Demonstrate payload string mutation."""
    print("\n" + "=" * 70)
    print("DEMO: Payload String Mutation")
    print("=" * 70)

    evolution = GeneticEvolution()

    # Original SQL injection payload
    original_payload = "' OR '1'='1"

    print("\n[1] Original Payload:")
    print("-" * 70)
    print(f"  {original_payload}")

    # Evolve the payload multiple times
    print("\n[2] Evolved Payloads:")
    print("-" * 70)
    for i in range(3):
        mutated, gene_seed, success = evolution.evolve_payload(original_payload, max_attempts=5)
        if success:
            print(f"  Generation {i + 1} (seed={gene_seed}):")
            print(f"    {repr(mutated)}")

    # Show evolution stats
    print("\n[3] Evolution Statistics:")
    print("-" * 70)
    stats = evolution.get_stats()
    print(f"  Total mutations: {stats['total']}")
    print(f"  Successful: {stats['successful']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")


def demo_polymorphic_engine():
    """Demonstrate the polymorphic engine directly."""
    print("\n" + "=" * 70)
    print("DEMO: Polymorphic Engine")
    print("=" * 70)

    from hive_zero_core.agents.genetic_evolution import PolymorphicEngine

    engine = PolymorphicEngine()

    original = "def hello(): return 'world'"

    print("\n[1] Same code, different signatures:")
    print("-" * 70)
    print(f"Original: {original}")

    # Generate 5 different mutations with different signatures
    for i in range(5):
        gene_seed = 1000 + i * 111
        mutated = engine.mutate_code(original, gene_seed)
        # Show first line to demonstrate signature difference
        first_line = mutated.split('\n')[0]
        print(f"  Seed {gene_seed}: {first_line}...")


def demo_natural_selection():
    """Demonstrate natural selection validation."""
    print("\n" + "=" * 70)
    print("DEMO: Natural Selection")
    print("=" * 70)

    from hive_zero_core.agents.genetic_evolution import NaturalSelection

    selector = NaturalSelection()

    test_cases = [
        ("def valid(): return 42", "Valid Python code"),
        ("x = 1 + 2\nprint(x)", "Valid Python script"),
        ("def broken(\nprint(", "Invalid syntax"),
        ("for x in", "Incomplete statement"),
    ]

    print("\n[1] Code Validation:")
    print("-" * 70)
    for code, description in test_cases:
        is_valid = selector.validate_python(code)
        status = "✓ PASS" if is_valid else "✗ FAIL"
        print(f"  {status}: {description}")
        print(f"         Code: {repr(code)}")


def main():
    """Run all demonstrations."""
    print("\n" + "#" * 70)
    print("# Adversarial-Swarm: Genetic Evolution Demo")
    print("# Red Team Capability Enhancement")
    print("#" * 70)

    try:
        demo_code_evolution()
        demo_payload_evolution()
        demo_polymorphic_engine()
        demo_natural_selection()

        print("\n" + "=" * 70)
        print("All demonstrations completed successfully!")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
