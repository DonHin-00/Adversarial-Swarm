# Skill Knowledge and Affects System

## Overview

HIVE-ZERO now includes an advanced skill knowledge system and expanded affect types that enable more sophisticated agent behavior, learning, and reward shaping.

## Skill System

### Skill Proficiency Levels

Agents now have explicit skill levels that affect their performance:

```python
from hive_zero_core.agents.base_expert import SkillLevel

# Available levels
SkillLevel.NOVICE      # 0.7x performance multiplier
SkillLevel.INTERMEDIATE # 1.0x performance multiplier  
SkillLevel.EXPERT      # 1.3x performance multiplier
SkillLevel.MASTER      # 1.5x performance multiplier
```

### Agent Skill Attributes

Each agent now tracks:
- **Primary Skills**: Core competencies
- **Secondary Skills**: Supporting abilities
- **Skill Statistics**: Activations, successes, failures, confidence
- **Effectiveness Score**: Calculated from success rate and confidence

### Example Usage

```python
from hive_zero_core.agents.recon_experts import CartographerAgent

cartographer = CartographerAgent(observation_dim=64, action_dim=64)
print(f"Skill Level: {cartographer.skill_level.name}")
print(f"Multiplier: {cartographer.get_skill_multiplier()}")
cartographer.record_activation(success=True, confidence=0.85)
```

## Threat Intelligence Knowledge Store

### Knowledge Base Structure

```python
from hive_zero_core.memory.threat_intelligence import ThreatIntelligenceStore

kb = ThreatIntelligenceStore()
recon_skills = kb.get_skills_by_category(ThreatCategory.RECONNAISSANCE)
kb.register_agent_knowledge("Cartographer", ["recon_001", "recon_002"])
```

## Advanced Reward Affects

### 8 Affect Types

1. **Adversarial** - Evasion ability
2. **Information** - Knowledge gain
3. **Stealth** - Traffic mimicry
4. **Temporal** - Speed/timing
5. **Resource** - Efficiency
6. **Reliability** - Consistency
7. **Novelty** - Discovery
8. **Coordination** - Teamwork

### Usage

```python
from hive_zero_core.training.advanced_rewards import AdvancedCompositeReward

reward_system = AdvancedCompositeReward()
rewards = reward_system.compute(
    adv_score=torch.tensor([0.85]),
    info_gain=0.3,
    actual_duration=2.5,
    target_duration=3.0,
    compute_cost=0.4,
    success_count=8,
    total_attempts=10,
    is_novel=True,
    active_agents=3
)
print(f"Total: {rewards['total']}")
```

## Benefits

✅ Granular agent proficiency tracking
✅ Structured knowledge taxonomy
✅ Multi-dimensional reward shaping
✅ Agent collaboration mechanisms
✅ Comprehensive performance metrics
