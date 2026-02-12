# Variant Breeding System

## Overview

The Variant Breeding System implements ephemeral agents with job-based lifecycles that:
- **Live only to complete assigned jobs** - then die and report intelligence
- **Scale with parent tier** - Higher tier parents produce stronger offspring with MORE JOBS
- **Cross-breed different roles** - Recon + Honeypot + WAF hybrids with trait blending
- **Feed intelligence back** - All learnings sent to central hub for collective growth
- **Specialize completely differently** - Each role has unique trait profiles

## Architecture

### Core Components

1. **Variant** - Ephemeral agent with role-specific specialization
2. **VariantJob** - Single task/mission for a variant
3. **IntelligenceHub** - Central aggregation point for all variant learnings
4. **VariantBreeder** - Spawns variants from parent units with tier-based scaling

### Variant Roles

Each role has **completely different specialization traits**:

#### RECONNAISSANCE
- `scan_speed`: How quickly targets are scanned
- `stealth_level`: Ability to avoid detection during scanning
- `coverage_breadth`: Network surface area covered
- `fingerprint_accuracy`: Precision of service identification

#### HONEYPOT
- `deception_level`: How convincing the trap appears
- `trap_sophistication`: Complexity of honeypot services
- `response_delay`: Timing patterns to waste attacker resources
- `intelligence_extraction`: Data gathering from attackers

#### WAF_BYPASS
- `evasion_creativity`: Novel bypass technique generation
- `encoding_depth`: Layers of obfuscation applied
- `pattern_breaking`: Ability to break signature detection
- `signature_mutation`: Payload polymorphism rate

#### STEALTH
- `footprint_minimization`: Reducing system traces
- `log_evasion`: Avoiding audit trails
- `memory_hiding`: Process concealment techniques
- `entropy_reduction`: Behavioral anomaly minimization

#### EXFILTRATION
- `bandwidth_efficiency`: Data transfer optimization
- `covert_channel_usage`: Hidden communication methods
- `data_compression`: Payload size reduction
- `protocol_mimicry`: Traffic disguise as legitimate

#### PERSISTENCE
- `hiding_effectiveness`: Implant concealment quality
- `resilience`: Resistance to removal attempts
- `reinfection_capability`: Auto-recovery mechanisms
- `dormancy_control`: Laying low periods

#### LATERAL_MOVEMENT
- `credential_harvesting`: Credential extraction rate
- `network_traversal`: Host-to-host movement ability
- `privilege_escalation`: Rights elevation techniques
- `host_enumeration`: Network mapping speed

#### PAYLOAD_GEN
- `exploit_potency`: Attack effectiveness
- `obfuscation_level`: Code hiding sophistication
- `polymorphism`: Signature mutation capability
- `stability`: Payload reliability

## Lifecycle

### 1. Birth (Breeding)

```python
# Single-role variant
variant = breeder.breed_variant(parent_unit, VariantRole.RECONNAISSANCE)

# Crosshybrid (50% more jobs!)
hybrid = breeder.cross_breed_variants(
    parent1, parent2,
    VariantRole.RECONNAISSANCE, VariantRole.WAF_BYPASS
)
```

**Job Count Scaling:**
- BASIC tier (0 merges): 1 job
- ENHANCED tier (1-2 merges): 2 jobs
- ADVANCED tier (3-5 merges): 3 jobs
- ELITE tier (6-10 merges): 5 jobs
- EXPERT tier (11-20 merges): 7 jobs
- MASTER tier (21+ merges): 10 jobs
- **Bonus**: +1 job per 3 merges
- **Cross-bred**: 50% more jobs (hybrid vigor!)

### 2. Job Assignment

```python
job = VariantJob(
    job_id="scan_001",
    job_type="scan_subnet",
    target="192.168.1.0/24",
    parameters={'ports': [80, 443, 22]}
)
variant.assign_job(job)
```

### 3. Job Execution & Completion

```python
# Simulate job execution
intelligence = {
    'hosts_discovered': 15,
    'open_ports': {'80': 10, '443': 8},
    'scan_duration': 32.5
}

variant.complete_job(job.job_id, intelligence, success=True)
```

### 4. Death & Intelligence Harvest

When all jobs complete, variant **automatically dies** and:
- Stops executing
- Generates final intelligence report
- Waits for harvesting

```python
breeder.harvest_intelligence(variant)
# → Intelligence sent to central hub
```

### 5. Intelligence Aggregation

Central hub collects all variant learnings:

```python
hub_intel = intelligence_hub.get_collective_intelligence(VariantRole.RECONNAISSANCE)
# Returns:
# - Total variants spawned/died
# - Success rates by role
# - Successful techniques
# - Pattern database
```

### 6. Inheritance

New variants inherit collective intelligence:

```python
# Next generation automatically boosted by past learnings
new_variant = breeder.breed_variant(parent, role)
# → fitness boosted by collective success rate
# → techniques informed by pattern database
```

## Tier-Based Offspring Quality

**THE MORE MERGES THE PARENT HAS, THE BETTER THE OFFSPRING:**

| Parent Tier | Parent Merges | Offspring Jobs | Offspring Fitness Boost | Specialization Multiplier |
|-------------|---------------|----------------|-------------------------|---------------------------|
| BASIC       | 0             | 1              | 1.0x                    | 1.0x                      |
| ENHANCED    | 1-2           | 2              | 1.1-1.2x                | 1.2x                      |
| ADVANCED    | 3-5           | 3              | 1.3-1.5x                | 1.4x                      |
| ELITE       | 6-10          | 5              | 1.6-2.0x                | 1.6x                      |
| EXPERT      | 11-20         | 7              | 2.1-3.0x                | 1.8x                      |
| MASTER      | 21+           | 10             | 3.1-5.0x                | 2.0x                      |

**Formula:**
```python
offspring_fitness = parent_fitness * (1.0 + parent_merge_count * 0.1)
trait_multiplier = 1.0 + (tier_value * 0.2)
```

## Cross-Breeding

Different role variants can merge to create **hybrids**:

```python
# Recon + WAF Bypass = Stealthy Scanner with Evasion
hybrid = breeder.cross_breed_variants(
    recon_parent, waf_parent,
    VariantRole.RECONNAISSANCE, VariantRole.WAF_BYPASS
)
```

**Hybrid Benefits:**
- **50% more jobs** than single-role variant
- **20% fitness boost** (hybrid vigor)
- **Blended specialization traits** from both roles
- **Dual intelligence injection** from both role hubs
- Primary role randomly selected

**Example Trait Blending:**
```
Recon Traits:           WAF Traits:             Hybrid Traits:
- scan_speed: 0.8       - evasion_creativity: 0.9   - scan_speed: 0.94 (+10%)
- stealth_level: 0.6    - encoding_depth: 0.7       - stealth_level: 0.72 (+10%)
                                                     - evasion_creativity: 0.99 (+10%)
                                                     - encoding_depth: 0.77 (+10%)
```

## Intelligence Flow

```
┌─────────────┐
│   Variant   │ (Recon, jobs=2)
└──────┬──────┘
       │ Execute Job 1: scan_subnet
       ├──> Intelligence: {hosts_discovered: 15, ...}
       │
       │ Execute Job 2: fingerprint_services
       ├──> Intelligence: {services: {...}, ...}
       │
       ▼ All jobs complete → DIE
┌──────────────┐
│   Harvest    │
└──────┬───────┘
       ▼
┌──────────────────────┐
│  Intelligence Hub     │ ← Aggregates from ALL dead variants
│  - Role statistics    │
│  - Success patterns   │
│  - Technique database │
└──────┬────────────────┘
       │ Collective Intelligence
       ▼
┌─────────────────┐
│ Next Generation │ ← Inherits learnings, starts stronger
└─────────────────┘
```

## Usage Examples

### Spawn Generation with Cross-Breeding

```python
hub = IntelligenceHub()
breeder = VariantBreeder(hub)

# Parent units at different tiers
parents = [basic_unit, enhanced_unit, elite_unit]

# Multiple roles
roles = [
    VariantRole.RECONNAISSANCE,
    VariantRole.HONEYPOT,
    VariantRole.WAF_BYPASS,
    VariantRole.STEALTH
]

# Spawn with 40% cross-breed rate
generation = breeder.spawn_generation(
    parents, roles,
    cross_breed_rate=0.4
)
# Result: ~60% single-role, ~40% hybrid variants
```

### Complete Lifecycle

```python
# 1. Breed
variant = breeder.breed_variant(elite_parent, VariantRole.RECONNAISSANCE)
# → 5 jobs (ELITE tier)

# 2. Assign jobs
for i in range(5):
    job = VariantJob(f"job_{i}", "scan_task", target=f"target_{i}")
    variant.assign_job(job)

# 3. Execute & complete
for job in variant.jobs:
    intelligence = execute_mission(job)  # Your execution logic
    variant.complete_job(job.job_id, intelligence, success=True)

# 4. Variant dies automatically after last job
assert not variant.is_alive

# 5. Harvest
breeder.harvest_intelligence(variant)

# 6. Intelligence now available for next generation
intel = hub.get_collective_intelligence(VariantRole.RECONNAISSANCE)
```

### Role Specialization Comparison

```python
# Create variants of different roles from same parent
recon_v = breeder.breed_variant(parent, VariantRole.RECONNAISSANCE)
honeypot_v = breeder.breed_variant(parent, VariantRole.HONEYPOT)
waf_v = breeder.breed_variant(parent, VariantRole.WAF_BYPASS)

print("RECONNAISSANCE traits:", recon_v.specialization_traits)
# → {scan_speed: 0.8, stealth_level: 0.6, ...}

print("HONEYPOT traits:", honeypot_v.specialization_traits)
# → {deception_level: 0.8, trap_sophistication: 0.7, ...}

print("WAF_BYPASS traits:", waf_v.specialization_traits)
# → {evasion_creativity: 0.8, encoding_depth: 0.7, ...}

# COMPLETELY DIFFERENT TRAITS!
```

## Breeding Statistics

Track breeding outcomes:

```python
stats = breeder.get_breeding_statistics()
```

Returns:
- `total_variants_bred`: Count of all variants created
- `cross_bred_variants`: Count of hybrid variants
- `cross_breed_rate`: Percentage of hybrids
- `average_success_rate`: Mean job success rate
- `tier_distribution`: Breakdown by tier

## Key Design Decisions

1. **Ephemeral by Design** - Variants don't persist, reducing attack surface
2. **Job-Based Termination** - Clear lifecycle prevents resource leaks
3. **Intelligence Feedback Loop** - Every death makes swarm smarter
4. **Tier-Based Scaling** - Natural progression from weak to strong
5. **Cross-Breeding** - Emergent combinations not explicitly programmed
6. **Role Specialization** - Prevents "one size fits all" agents

## Integration with Existing System

Variant breeding integrates with:
- **SwarmFusion**: Parents are SwarmUnits from fusion system
- **CapabilityEscalation**: Tier system determines offspring strength
- **GeneticEvolution**: Genome mutations apply to variants
- **IntelligenceHub**: Central point in synergistic system

## Security Considerations

For **authorized red team operations only**:
- Educational and training purposes
- Controlled penetration testing environments
- Security research with proper authorization

Variants implement real techniques:
- Port scanning (reconnaissance)
- Honeypot deployment (defensive)
- WAF evasion (offensive testing)
- Lateral movement (breach simulation)

---

**Remember: The more merges, the more jobs, the stronger the offspring!**
