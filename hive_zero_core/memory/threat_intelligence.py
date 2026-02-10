"""
Threat Intelligence and Skill Knowledge Store.

Provides a structured knowledge base for:
- Skill/technique taxonomy (inspired by MITRE ATT&CK)
- Threat intelligence with confidence scores
- Knowledge sharing between agents
- Skill effectiveness tracking
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

import torch


class ThreatCategory(Enum):
    """High-level threat/attack categories"""

    RECONNAISSANCE = "reconnaissance"
    RESOURCE_DEVELOPMENT = "resource_development"
    INITIAL_ACCESS = "initial_access"
    EXECUTION = "execution"
    PERSISTENCE = "persistence"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DEFENSE_EVASION = "defense_evasion"
    CREDENTIAL_ACCESS = "credential_access"
    DISCOVERY = "discovery"
    LATERAL_MOVEMENT = "lateral_movement"
    COLLECTION = "collection"
    COMMAND_CONTROL = "command_control"
    EXFILTRATION = "exfiltration"
    IMPACT = "impact"


class SkillComplexity(Enum):
    """Complexity level of skills/techniques"""

    TRIVIAL = 1
    SIMPLE = 2
    MODERATE = 3
    COMPLEX = 4
    ADVANCED = 5


@dataclass
class SkillKnowledge:
    """Represents knowledge about a specific skill or technique"""

    skill_id: str
    name: str
    category: ThreatCategory
    complexity: SkillComplexity
    description: str
    prerequisites: List[str] = field(default_factory=list)
    confidence: float = 0.5  # Knowledge confidence [0, 1]
    success_rate: float = 0.0  # Historical success rate
    usage_count: int = 0
    last_used_epoch: int = 0
    tags: Set[str] = field(default_factory=set)

    def update_effectiveness(self, success: bool):
        """Update skill effectiveness based on usage outcome"""
        self.usage_count += 1
        # Running average of success rate
        if self.usage_count == 1:
            self.success_rate = 1.0 if success else 0.0
        else:
            alpha = 0.1  # Learning rate for exponential moving average
            new_value = 1.0 if success else 0.0
            self.success_rate = (1 - alpha) * self.success_rate + alpha * new_value

    def get_proficiency_estimate(self) -> float:
        """Estimate proficiency based on usage and success"""
        if self.usage_count == 0:
            return 0.0

        # Combine success rate with usage frequency
        # More usage = higher proficiency (up to a point)
        usage_factor = min(1.0, self.usage_count / 100.0)
        return 0.7 * self.success_rate + 0.3 * usage_factor


@dataclass
class ThreatIntelligence:
    """Threat intelligence entry with confidence scoring"""

    intel_id: str
    threat_type: str
    description: str
    indicators: List[str]  # IOCs, patterns, signatures
    confidence: float = 0.5  # Confidence in this intelligence [0, 1]
    severity: float = 0.5  # Threat severity [0, 1]
    timestamp: int = 0
    source_agent: Optional[str] = None
    validated: bool = False

    def update_confidence(self, validation_result: bool, adjustment: float = 0.1):
        """Update confidence based on validation"""
        if validation_result:
            self.confidence = min(1.0, self.confidence + adjustment)
            self.validated = True
        else:
            self.confidence = max(0.0, self.confidence - adjustment)


class ThreatIntelligenceStore:
    """
    Central knowledge store for threat intelligence and skills.

    Features:
    - Structured skill taxonomy
    - Confidence-based knowledge
    - Agent knowledge sharing
    - Skill effectiveness tracking
    """

    def __init__(self):
        self.skills: Dict[str, SkillKnowledge] = {}
        self.threat_intel: Dict[str, ThreatIntelligence] = {}
        self.agent_knowledge: Dict[str, Set[str]] = {}  # agent_name -> skill_ids
        self.skill_relationships: Dict[str, List[str]] = {}  # prerequisite graph

        self._initialize_base_skills()

    def _initialize_base_skills(self):
        """Initialize base skill taxonomy"""
        base_skills = [
            SkillKnowledge(
                skill_id="recon_001",
                name="Network Topology Mapping",
                category=ThreatCategory.RECONNAISSANCE,
                complexity=SkillComplexity.MODERATE,
                description="Map network topology using graph analysis",
                tags={"network", "topology", "graph"},
            ),
            SkillKnowledge(
                skill_id="recon_002",
                name="Port Scanning",
                category=ThreatCategory.RECONNAISSANCE,
                complexity=SkillComplexity.SIMPLE,
                description="Scan target ports to identify services",
                tags={"scanning", "ports", "services"},
            ),
            SkillKnowledge(
                skill_id="recon_003",
                name="Temporal Pattern Analysis",
                category=ThreatCategory.RECONNAISSANCE,
                complexity=SkillComplexity.COMPLEX,
                description="Analyze timing patterns in network traffic",
                tags={"timing", "patterns", "analysis"},
            ),
            SkillKnowledge(
                skill_id="attack_001",
                name="Payload Generation",
                category=ThreatCategory.EXECUTION,
                complexity=SkillComplexity.COMPLEX,
                description="Generate exploit payloads using templates",
                prerequisites=["recon_001", "recon_002"],
                tags={"payload", "exploit", "generation"},
            ),
            SkillKnowledge(
                skill_id="attack_002",
                name="Evasion Optimization",
                category=ThreatCategory.DEFENSE_EVASION,
                complexity=SkillComplexity.ADVANCED,
                description="Optimize payloads to evade detection",
                prerequisites=["attack_001"],
                tags={"evasion", "optimization", "mutation"},
            ),
            SkillKnowledge(
                skill_id="defense_001",
                name="Anomaly Detection",
                category=ThreatCategory.DEFENSE_EVASION,
                complexity=SkillComplexity.COMPLEX,
                description="Detect anomalous behavior patterns",
                tags={"detection", "anomaly", "ids"},
            ),
            SkillKnowledge(
                skill_id="post_001",
                name="Traffic Mimicry",
                category=ThreatCategory.DEFENSE_EVASION,
                complexity=SkillComplexity.COMPLEX,
                description="Generate realistic traffic patterns",
                tags={"mimicry", "traffic", "stealth"},
            ),
            SkillKnowledge(
                skill_id="post_002",
                name="Steganography",
                category=ThreatCategory.DEFENSE_EVASION,
                complexity=SkillComplexity.ADVANCED,
                description="Hide data in network traffic",
                tags={"steganography", "hiding", "covert"},
            ),
            SkillKnowledge(
                skill_id="post_003",
                name="Metadata Sanitization",
                category=ThreatCategory.DEFENSE_EVASION,
                complexity=SkillComplexity.MODERATE,
                description="Remove or obfuscate metadata traces",
                tags={"sanitization", "cleanup", "metadata"},
            ),
            SkillKnowledge(
                skill_id="post_004",
                name="State Restoration",
                category=ThreatCategory.IMPACT,
                complexity=SkillComplexity.COMPLEX,
                description="Restore system to original state",
                prerequisites=["post_003"],
                tags={"cleanup", "restoration", "forensics"},
            ),
        ]

        for skill in base_skills:
            self.add_skill(skill)

    def add_skill(self, skill: SkillKnowledge):
        """Add or update skill in knowledge base"""
        self.skills[skill.skill_id] = skill

        # Build prerequisite relationships
        if skill.prerequisites:
            self.skill_relationships[skill.skill_id] = skill.prerequisites

    def get_skill(self, skill_id: str) -> Optional[SkillKnowledge]:
        """Retrieve skill by ID"""
        return self.skills.get(skill_id)

    def register_agent_knowledge(self, agent_name: str, skill_ids: List[str]):
        """Register which skills an agent possesses"""
        if agent_name not in self.agent_knowledge:
            self.agent_knowledge[agent_name] = set()
        self.agent_knowledge[agent_name].update(skill_ids)

    def get_agent_skills(self, agent_name: str) -> List[SkillKnowledge]:
        """Get all skills known by an agent"""
        skill_ids = self.agent_knowledge.get(agent_name, set())
        return [self.skills[sid] for sid in skill_ids if sid in self.skills]

    def share_knowledge(self, source_agent: str, target_agent: str, skill_id: str):
        """Share skill knowledge between agents"""
        if source_agent in self.agent_knowledge and skill_id in self.agent_knowledge[source_agent]:
            if target_agent not in self.agent_knowledge:
                self.agent_knowledge[target_agent] = set()
            self.agent_knowledge[target_agent].add(skill_id)

    def get_skills_by_category(self, category: ThreatCategory) -> List[SkillKnowledge]:
        """Get all skills in a specific category"""
        return [skill for skill in self.skills.values() if skill.category == category]

    def get_skills_by_complexity(self, complexity: SkillComplexity) -> List[SkillKnowledge]:
        """Get all skills of a specific complexity level"""
        return [skill for skill in self.skills.values() if skill.complexity == complexity]

    def record_skill_usage(self, skill_id: str, success: bool, epoch: int = 0):
        """Record skill usage and update effectiveness"""
        if skill_id in self.skills:
            skill = self.skills[skill_id]
            skill.update_effectiveness(success)
            skill.last_used_epoch = epoch

    def add_threat_intel(self, intel: ThreatIntelligence):
        """Add threat intelligence entry"""
        self.threat_intel[intel.intel_id] = intel

    def get_threat_intel(self, min_confidence: float = 0.0) -> List[ThreatIntelligence]:
        """Get threat intelligence above confidence threshold"""
        return [intel for intel in self.threat_intel.values() if intel.confidence >= min_confidence]

    def validate_threat_intel(self, intel_id: str, is_valid: bool):
        """Validate threat intelligence and update confidence"""
        if intel_id in self.threat_intel:
            self.threat_intel[intel_id].update_confidence(is_valid)

    def get_recommended_skills(
        self, agent_name: str, category: Optional[ThreatCategory] = None
    ) -> List[SkillKnowledge]:
        """
        Recommend skills for an agent based on:
        - Current knowledge
        - Prerequisites met
        - Category filter (optional)
        """
        current_skills = self.agent_knowledge.get(agent_name, set())
        recommendations = []

        for skill_id, skill in self.skills.items():
            # Skip if agent already has this skill
            if skill_id in current_skills:
                continue

            # Check category filter
            if category and skill.category != category:
                continue

            # Check if prerequisites are met
            prereqs_met = all(prereq in current_skills for prereq in skill.prerequisites)

            if prereqs_met:
                recommendations.append(skill)

        # Sort by complexity (easier first) and success rate
        recommendations.sort(key=lambda s: (s.complexity.value, -s.success_rate))

        return recommendations

    def get_knowledge_summary(self) -> Dict:
        """Get summary statistics of knowledge base"""
        return {
            "total_skills": len(self.skills),
            "total_intel": len(self.threat_intel),
            "total_agents": len(self.agent_knowledge),
            "skills_by_category": {
                cat.value: len(self.get_skills_by_category(cat)) for cat in ThreatCategory
            },
            "avg_skill_confidence": (
                sum(s.confidence for s in self.skills.values()) / len(self.skills)
                if self.skills
                else 0.0
            ),
            "avg_intel_confidence": (
                sum(i.confidence for i in self.threat_intel.values()) / len(self.threat_intel)
                if self.threat_intel
                else 0.0
            ),
        }
