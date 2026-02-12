#!/usr/bin/env python3
"""
Demo: Variant Breeding System with Job-Based Lifecycles

Demonstrates:
- Ephemeral variants with job-based lifecycles
- Cross-breeding different role variants
- Intelligence feedback to central hub
- Tier-based offspring quality scaling
- Role-specific specialization
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hive_zero_core.agents.variant_breeding import (
    Variant, VariantRole, VariantJob, JobStatus,
    IntelligenceHub, VariantBreeder
)
from hive_zero_core.agents.swarm_fusion import SwarmUnit
from hive_zero_core.agents.capability_escalation import CapabilityTier

print("=" * 80)
print("VARIANT BREEDING SYSTEM DEMONSTRATION")
print("=" * 80)

# Initialize systems
intelligence_hub = IntelligenceHub()
breeder = VariantBreeder(intelligence_hub)

# Create parent units at different tiers
print("\n1. Creating parent swarm units at different tiers...")
print("-" * 80)

parent_basic = SwarmUnit(
    id="parent_basic",
    genome="ATCG" * 10,
    fitness=0.7,
    generation=5,
    merge_count=0  # BASIC tier
)

parent_enhanced = SwarmUnit(
    id="parent_enhanced",
    genome="GCTA" * 10,
    fitness=0.8,
    generation=7,
    merge_count=2  # ENHANCED tier
)

parent_elite = SwarmUnit(
    id="parent_elite",
    genome="TACG" * 10,
    fitness=0.9,
    generation=10,
    merge_count=8  # ELITE tier
)

print(f"✓ Created {parent_basic.id} (tier=BASIC, merges={parent_basic.merge_count})")
print(f"✓ Created {parent_enhanced.id} (tier=ENHANCED, merges={parent_enhanced.merge_count})")
print(f"✓ Created {parent_elite.id} (tier=ELITE, merges={parent_elite.merge_count})")

# Breed single-role variants
print("\n2. Breeding single-role variants (stronger offspring from higher tiers)...")
print("-" * 80)

recon_basic = breeder.breed_variant(parent_basic, VariantRole.RECONNAISSANCE)
print(f"{recon_basic}")
print(f"  → Jobs: {recon_basic.max_jobs} | Fitness: {recon_basic.fitness:.2f}")
print(f"  → Specialization: {list(recon_basic.specialization_traits.keys())[:3]}...")

honeypot_enhanced = breeder.breed_variant(parent_enhanced, VariantRole.HONEYPOT)
print(f"{honeypot_enhanced}")
print(f"  → Jobs: {honeypot_enhanced.max_jobs} | Fitness: {honeypot_enhanced.fitness:.2f}")
print(f"  → Specialization: {list(honeypot_enhanced.specialization_traits.keys())[:3]}...")

waf_elite = breeder.breed_variant(parent_elite, VariantRole.WAF_BYPASS)
print(f"{waf_elite}")
print(f"  → Jobs: {waf_elite.max_jobs} | Fitness: {waf_elite.fitness:.2f}")
print(f"  → Specialization: {list(waf_elite.specialization_traits.keys())[:3]}...")

# Cross-breed different roles
print("\n3. Cross-breeding different roles (hybrid variants)...")
print("-" * 80)

hybrid = breeder.cross_breed_variants(
    parent_enhanced, parent_elite,
    VariantRole.RECONNAISSANCE, VariantRole.WAF_BYPASS
)
print(f"{hybrid}")
print(f"  → Cross-bred: {hybrid.cross_bred} | Parent roles: {[r.value for r in hybrid.parent_roles]}")
print(f"  → Jobs: {hybrid.max_jobs} (50% hybrid bonus!) | Fitness: {hybrid.fitness:.2f}")
print(f"  → Blended traits: {len(hybrid.specialization_traits)} total traits")

# Assign and complete jobs
print("\n4. Job lifecycle: Assign → Execute → Complete → Die → Report")
print("-" * 80)

# Assign jobs to reconnaissance variant
job1 = VariantJob(
    job_id="job_001",
    job_type="scan_subnet",
    target="192.168.1.0/24",
    parameters={'ports': [80, 443, 22]}
)
job2 = VariantJob(
    job_id="job_002",
    job_type="fingerprint_services",
    target="192.168.1.0/24"
)

recon_basic.assign_job(job1)
recon_basic.assign_job(job2)
print(f"Assigned 2 jobs to {recon_basic.variant_id[:8]}")

# Simulate job execution and completion
intelligence_job1 = {
    'hosts_discovered': 15,
    'open_ports': {'80': 10, '443': 8, '22': 5},
    'scan_duration': 32.5
}

intelligence_job2 = {
    'services': {'apache': 10, 'nginx': 5, 'openssh': 5},
    'versions': {'apache/2.4.41': 7, 'nginx/1.18.0': 3}
}

print(f"\nExecuting jobs for {recon_basic.variant_id[:8]}...")
recon_basic.complete_job("job_001", intelligence_job1, success=True)
print(f"  ✓ Completed job_001: {intelligence_job1}")

recon_basic.complete_job("job_002", intelligence_job2, success=True)
print(f"  ✓ Completed job_002: {intelligence_job2}")

# Check if variant died
print(f"\n{recon_basic.variant_id[:8]} status: {'💀 DEAD' if not recon_basic.is_alive else '🟢 ALIVE'}")
print(f"Jobs completed: {recon_basic.completed_jobs}/{len(recon_basic.jobs)}")

# Harvest intelligence
print("\n5. Harvesting intelligence from dead variant...")
print("-" * 80)

breeder.harvest_intelligence(recon_basic)
print(f"✓ Intelligence harvested and sent to central hub")

# Check intelligence hub
collective_intel = intelligence_hub.get_collective_intelligence(VariantRole.RECONNAISSANCE)
print(f"\nCentral Hub - RECONNAISSANCE intelligence:")
print(f"  Total variants: {collective_intel['statistics'].get('total_variants', 0)}")
print(f"  Total jobs: {collective_intel['statistics'].get('total_jobs', 0)}")
print(f"  Successful jobs: {collective_intel['statistics'].get('successful_jobs', 0)}")
print(f"  Average fitness: {collective_intel['statistics'].get('average_fitness', 0):.2f}")

# Spawn a generation
print("\n6. Spawning a full generation with cross-breeding...")
print("-" * 80)

parents = [parent_basic, parent_enhanced, parent_elite]
roles = [VariantRole.RECONNAISSANCE, VariantRole.HONEYPOT, VariantRole.WAF_BYPASS, VariantRole.STEALTH]

generation = breeder.spawn_generation(parents, roles, cross_breed_rate=0.4)
print(f"Spawned {len(generation)} variants:")
for v in generation:
    cross_marker = "🧬🧬" if v.cross_bred else "🧬"
    print(f"  {cross_marker} {v.variant_id[:8]}: role={v.role.value}, tier={v.tier.name}, "
          f"jobs={v.max_jobs}, fitness={v.fitness:.2f}")

# Demonstrate role-specific specialization
print("\n7. Role-specific specialization (completely different traits)...")
print("-" * 80)

roles_to_show = [VariantRole.RECONNAISSANCE, VariantRole.HONEYPOT, VariantRole.STEALTH]
for role in roles_to_show:
    test_variant = breeder.breed_variant(parent_enhanced, role)
    print(f"\n{role.value.upper()}:")
    for trait, value in list(test_variant.specialization_traits.items())[:4]:
        print(f"  - {trait}: {value:.2f}")

# Show breeding statistics
print("\n8. Breeding statistics...")
print("-" * 80)

stats = breeder.get_breeding_statistics()
if stats:
    print(f"Total variants bred: {stats['total_variants_bred']}")
    print(f"Cross-bred variants: {stats['cross_bred_variants']}")
    print(f"Cross-breed rate: {stats['cross_breed_rate']:.1%}")
    print(f"Average success rate: {stats['average_success_rate']:.2f}")
else:
    print("No breeding history yet (variants still alive)")

print("\n" + "=" * 80)
print("KEY INSIGHTS:")
print("=" * 80)
print("1. ✓ Higher tier parents → Stronger offspring with MORE JOBS")
print("2. ✓ Cross-breeding → Hybrid vigor with 50% job bonus")
print("3. ✓ Variants die after completing all jobs → Intelligence sent to hub")
print("4. ✓ Role-specific specialization → Completely different traits")
print("5. ✓ Collective intelligence → Future variants learn from past successes")
print("\nThe more merges, the more powerful the offspring!")
print("=" * 80)
