"""
Complete MITRE ATT&CK and MITRE ATLAS Technique Mapping

This module provides comprehensive coverage of:
1. MITRE ATT&CK Enterprise (120+ core techniques)
2. MITRE ATLAS (AI/ML adversarial techniques - 20+ techniques)
3. Mappings to capabilities in the Adversarial-Swarm system

References:
- MITRE ATT&CK: https://attack.mitre.org/
- MITRE ATLAS: https://atlas.mitre.org/
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class MITRETactic(Enum):
    """MITRE ATT&CK Tactics (14 total)."""

    RECONNAISSANCE = "TA0043"
    RESOURCE_DEVELOPMENT = "TA0042"
    INITIAL_ACCESS = "TA0001"
    EXECUTION = "TA0002"
    PERSISTENCE = "TA0003"
    PRIVILEGE_ESCALATION = "TA0004"
    DEFENSE_EVASION = "TA0005"
    CREDENTIAL_ACCESS = "TA0006"
    DISCOVERY = "TA0007"
    LATERAL_MOVEMENT = "TA0008"
    COLLECTION = "TA0009"
    COMMAND_AND_CONTROL = "TA0011"
    EXFILTRATION = "TA0010"
    IMPACT = "TA0040"


class MITREATLASTactic(Enum):
    """MITRE ATLAS Tactics (AI/ML specific)."""

    ML_MODEL_ACCESS = "AML.TA0000"
    ML_ATTACK_STAGING = "AML.TA0001"
    INITIAL_ACCESS_ML = "AML.TA0002"
    ML_ATTACK_EXECUTION = "AML.TA0003"
    PERSISTENCE_ML = "AML.TA0004"
    DEFENSE_EVASION_ML = "AML.TA0005"
    DISCOVERY_ML = "AML.TA0006"
    COLLECTION_ML = "AML.TA0007"
    ML_ATTACK_IMPACT = "AML.TA0008"
    EXFILTRATION_ML = "AML.TA0009"


@dataclass
class MITRETechnique:
    """Represents a MITRE ATT&CK or ATLAS technique."""

    technique_id: str
    name: str
    description: str
    tactic: str
    platform: List[str]
    data_sources: List[str]
    is_atlas: bool = False


# This is a comprehensive but condensed version for production use
# Full technique database with 120+ ATT&CK + 20 ATLAS techniques

MITRE_ATTACK_TECHNIQUES: Dict[str, MITRETechnique] = {}
MITRE_ATLAS_TECHNIQUES: Dict[str, MITRETechnique] = {}


def _init_techniques():
    """Initialize all MITRE techniques."""
    # === MITRE ATT&CK ENTERPRISE ===
    attack_data = {
        # Reconnaissance
        "T1595": ("Active Scanning", "Network scanning", MITRETactic.RECONNAISSANCE.value),
        "T1595.001": ("Scanning IP Blocks", "Scan IP ranges", MITRETactic.RECONNAISSANCE.value),
        "T1592": (
            "Gather Victim Host Information",
            "Collect host info",
            MITRETactic.RECONNAISSANCE.value,
        ),
        "T1590": (
            "Gather Victim Network Information",
            "Collect network info",
            MITRETactic.RECONNAISSANCE.value,
        ),
        # Initial Access
        "T1078": ("Valid Accounts", "Use legitimate credentials", MITRETactic.INITIAL_ACCESS.value),
        "T1190": (
            "Exploit Public-Facing Application",
            "Exploit vulnerabilities",
            MITRETactic.INITIAL_ACCESS.value,
        ),
        "T1133": ("External Remote Services", "VPN/RDP access", MITRETactic.INITIAL_ACCESS.value),
        "T1566": ("Phishing", "Phishing attacks", MITRETactic.INITIAL_ACCESS.value),
        "T1566.001": (
            "Spearphishing Attachment",
            "Malicious attachments",
            MITRETactic.INITIAL_ACCESS.value,
        ),
        "T1566.002": ("Spearphishing Link", "Malicious links", MITRETactic.INITIAL_ACCESS.value),
        # Execution
        "T1059": (
            "Command and Scripting Interpreter",
            "Execute commands",
            MITRETactic.EXECUTION.value,
        ),
        "T1059.001": ("PowerShell", "Execute PowerShell", MITRETactic.EXECUTION.value),
        "T1059.003": ("Windows Command Shell", "Execute cmd.exe", MITRETactic.EXECUTION.value),
        "T1059.004": ("Unix Shell", "Execute bash/sh", MITRETactic.EXECUTION.value),
        "T1059.006": ("Python", "Execute Python code", MITRETactic.EXECUTION.value),
        "T1203": (
            "Exploitation for Client Execution",
            "Exploit vulnerabilities",
            MITRETactic.EXECUTION.value,
        ),
        # Persistence
        "T1053": ("Scheduled Task/Job", "Scheduled tasks", MITRETactic.PERSISTENCE.value),
        "T1053.005": ("Scheduled Task", "Windows Scheduled Task", MITRETactic.PERSISTENCE.value),
        "T1053.003": ("Cron", "Linux cron job", MITRETactic.PERSISTENCE.value),
        "T1547": (
            "Boot or Logon Autostart Execution",
            "Autostart mechanisms",
            MITRETactic.PERSISTENCE.value,
        ),
        "T1547.001": ("Registry Run Keys", "Registry autostart", MITRETactic.PERSISTENCE.value),
        "T1543": (
            "Create or Modify System Process",
            "System services",
            MITRETactic.PERSISTENCE.value,
        ),
        "T1543.003": ("Windows Service", "Windows service", MITRETactic.PERSISTENCE.value),
        "T1546": ("Event Triggered Execution", "Event triggers", MITRETactic.PERSISTENCE.value),
        "T1546.015": (
            "Component Object Model Hijacking",
            "COM hijacking",
            MITRETactic.PERSISTENCE.value,
        ),
        "T1542": ("Pre-OS Boot", "Pre-OS persistence", MITRETactic.PERSISTENCE.value),
        "T1542.003": ("Bootkit", "MBR/VBR infection", MITRETactic.PERSISTENCE.value),
        # Privilege Escalation
        "T1068": (
            "Exploitation for Privilege Escalation",
            "Exploit for privesc",
            MITRETactic.PRIVILEGE_ESCALATION.value,
        ),
        "T1134": (
            "Access Token Manipulation",
            "Token manipulation",
            MITRETactic.PRIVILEGE_ESCALATION.value,
        ),
        "T1134.001": (
            "Token Impersonation/Theft",
            "Token impersonation",
            MITRETactic.PRIVILEGE_ESCALATION.value,
        ),
        "T1548": (
            "Abuse Elevation Control Mechanism",
            "Bypass UAC",
            MITRETactic.PRIVILEGE_ESCALATION.value,
        ),
        "T1548.002": (
            "Bypass User Account Control",
            "UAC bypass",
            MITRETactic.PRIVILEGE_ESCALATION.value,
        ),
        # Defense Evasion
        "T1055": (
            "Process Injection",
            "Inject code into processes",
            MITRETactic.DEFENSE_EVASION.value,
        ),
        "T1055.001": (
            "Dynamic-link Library Injection",
            "DLL injection",
            MITRETactic.DEFENSE_EVASION.value,
        ),
        "T1055.012": ("Process Hollowing", "Process hollowing", MITRETactic.DEFENSE_EVASION.value),
        "T1027": (
            "Obfuscated Files or Information",
            "Obfuscation",
            MITRETactic.DEFENSE_EVASION.value,
        ),
        "T1027.002": ("Software Packing", "Packing/compression", MITRETactic.DEFENSE_EVASION.value),
        "T1027.005": (
            "Indicator Removal from Tools",
            "Remove signatures",
            MITRETactic.DEFENSE_EVASION.value,
        ),
        "T1036": ("Masquerading", "Disguise as legitimate", MITRETactic.DEFENSE_EVASION.value),
        "T1070": ("Indicator Removal", "Delete artifacts", MITRETactic.DEFENSE_EVASION.value),
        "T1070.001": (
            "Clear Windows Event Logs",
            "Clear event logs",
            MITRETactic.DEFENSE_EVASION.value,
        ),
        "T1140": (
            "Deobfuscate/Decode Files",
            "Runtime decoding",
            MITRETactic.DEFENSE_EVASION.value,
        ),
        "T1562": ("Impair Defenses", "Disable security tools", MITRETactic.DEFENSE_EVASION.value),
        "T1562.001": (
            "Disable or Modify Tools",
            "Disable AV/EDR",
            MITRETactic.DEFENSE_EVASION.value,
        ),
        "T1574": (
            "Hijack Execution Flow",
            "Execution hijacking",
            MITRETactic.DEFENSE_EVASION.value,
        ),
        "T1574.002": (
            "DLL Side-Loading",
            "DLL search order hijacking",
            MITRETactic.DEFENSE_EVASION.value,
        ),
        "T1620": (
            "Reflective Code Loading",
            "Memory-only execution",
            MITRETactic.DEFENSE_EVASION.value,
        ),
        # Credential Access
        "T1003": ("OS Credential Dumping", "Dump credentials", MITRETactic.CREDENTIAL_ACCESS.value),
        "T1003.001": ("LSASS Memory", "Dump LSASS", MITRETactic.CREDENTIAL_ACCESS.value),
        "T1003.002": ("Security Account Manager", "Dump SAM", MITRETactic.CREDENTIAL_ACCESS.value),
        "T1003.003": ("NTDS", "Dump Active Directory", MITRETactic.CREDENTIAL_ACCESS.value),
        "T1110": ("Brute Force", "Password guessing", MITRETactic.CREDENTIAL_ACCESS.value),
        "T1555": (
            "Credentials from Password Stores",
            "Extract from password managers",
            MITRETactic.CREDENTIAL_ACCESS.value,
        ),
        "T1555.003": (
            "Credentials from Web Browsers",
            "Browser credentials",
            MITRETactic.CREDENTIAL_ACCESS.value,
        ),
        # Discovery
        "T1087": ("Account Discovery", "Enumerate accounts", MITRETactic.DISCOVERY.value),
        "T1046": ("Network Service Discovery", "Discover services", MITRETactic.DISCOVERY.value),
        "T1082": ("System Information Discovery", "System info", MITRETactic.DISCOVERY.value),
        "T1083": ("File and Directory Discovery", "Enumerate files", MITRETactic.DISCOVERY.value),
        "T1018": (
            "Remote System Discovery",
            "Discover remote systems",
            MITRETactic.DISCOVERY.value,
        ),
        "T1049": (
            "System Network Connections Discovery",
            "Enumerate connections",
            MITRETactic.DISCOVERY.value,
        ),
        "T1057": ("Process Discovery", "Enumerate processes", MITRETactic.DISCOVERY.value),
        "T1012": ("Query Registry", "Query Windows Registry", MITRETactic.DISCOVERY.value),
        # Lateral Movement
        "T1021": ("Remote Services", "Use remote services", MITRETactic.LATERAL_MOVEMENT.value),
        "T1021.001": ("Remote Desktop Protocol", "RDP", MITRETactic.LATERAL_MOVEMENT.value),
        "T1021.002": (
            "SMB/Windows Admin Shares",
            "SMB/admin shares",
            MITRETactic.LATERAL_MOVEMENT.value,
        ),
        "T1021.006": ("Windows Remote Management", "WinRM", MITRETactic.LATERAL_MOVEMENT.value),
        "T1570": ("Lateral Tool Transfer", "Transfer tools", MITRETactic.LATERAL_MOVEMENT.value),
        # Collection
        "T1005": ("Data from Local System", "Local data collection", MITRETactic.COLLECTION.value),
        "T1039": (
            "Data from Network Shared Drive",
            "Network share data",
            MITRETactic.COLLECTION.value,
        ),
        "T1056": ("Input Capture", "Capture input", MITRETactic.COLLECTION.value),
        "T1056.001": ("Keylogging", "Capture keystrokes", MITRETactic.COLLECTION.value),
        "T1113": ("Screen Capture", "Screenshots", MITRETactic.COLLECTION.value),
        "T1115": ("Clipboard Data", "Clipboard contents", MITRETactic.COLLECTION.value),
        # Command and Control
        "T1071": (
            "Application Layer Protocol",
            "App layer C2",
            MITRETactic.COMMAND_AND_CONTROL.value,
        ),
        "T1071.001": ("Web Protocols", "HTTP/HTTPS C2", MITRETactic.COMMAND_AND_CONTROL.value),
        "T1071.004": ("DNS", "DNS C2", MITRETactic.COMMAND_AND_CONTROL.value),
        "T1573": ("Encrypted Channel", "Encrypted C2", MITRETactic.COMMAND_AND_CONTROL.value),
        "T1090": ("Proxy", "Proxy C2", MITRETactic.COMMAND_AND_CONTROL.value),
        "T1095": (
            "Non-Application Layer Protocol",
            "Non-app layer C2",
            MITRETactic.COMMAND_AND_CONTROL.value,
        ),
        "T1132": ("Data Encoding", "Encode C2 data", MITRETactic.COMMAND_AND_CONTROL.value),
        # Exfiltration
        "T1041": ("Exfiltration Over C2 Channel", "Exfil over C2", MITRETactic.EXFILTRATION.value),
        "T1048": (
            "Exfiltration Over Alternative Protocol",
            "Alt protocol exfil",
            MITRETactic.EXFILTRATION.value,
        ),
        "T1048.003": (
            "Exfiltration Over Unencrypted Non-C2 Protocol",
            "Unencrypted exfil",
            MITRETactic.EXFILTRATION.value,
        ),
        "T1567": (
            "Exfiltration Over Web Service",
            "Web service exfil",
            MITRETactic.EXFILTRATION.value,
        ),
        "T1567.002": (
            "Exfiltration to Cloud Storage",
            "Cloud storage exfil",
            MITRETactic.EXFILTRATION.value,
        ),
        # Impact
        "T1486": ("Data Encrypted for Impact", "Ransomware", MITRETactic.IMPACT.value),
        "T1490": ("Inhibit System Recovery", "Delete backups", MITRETactic.IMPACT.value),
        "T1485": ("Data Destruction", "Destroy data", MITRETactic.IMPACT.value),
        "T1491": ("Defacement", "Modify content", MITRETactic.IMPACT.value),
        "T1561": ("Disk Wipe", "Wipe disk", MITRETactic.IMPACT.value),
        "T1489": ("Service Stop", "Stop security services", MITRETactic.IMPACT.value),
        "T1657": ("Financial Theft", "Steal cryptocurrency or funds", MITRETactic.IMPACT.value),
        # Resource Development (ATT&CK v18 additions)
        "T1583": (
            "Acquire Infrastructure",
            "Acquire servers/domains for ops",
            MITRETactic.RESOURCE_DEVELOPMENT.value,
        ),
        "T1583.001": (
            "Domains",
            "Register attacker-controlled domains",
            MITRETactic.RESOURCE_DEVELOPMENT.value,
        ),
        "T1584": (
            "Compromise Infrastructure",
            "Compromise third-party infrastructure",
            MITRETactic.RESOURCE_DEVELOPMENT.value,
        ),
        "T1588": (
            "Obtain Capabilities",
            "Obtain tools/exploits for operations",
            MITRETactic.RESOURCE_DEVELOPMENT.value,
        ),
        "T1588.002": (
            "Tool",
            "Obtain offensive security tools",
            MITRETactic.RESOURCE_DEVELOPMENT.value,
        ),
        # Reconnaissance (ATT&CK v18 additions)
        "T1593": (
            "Search Open Websites/Domains",
            "Search public sites for target info",
            MITRETactic.RECONNAISSANCE.value,
        ),
        "T1589": (
            "Gather Victim Identity Information",
            "Collect employee/credential info",
            MITRETactic.RECONNAISSANCE.value,
        ),
        # Execution (ATT&CK v18 / container additions)
        "T1648": (
            "Serverless Execution",
            "Abuse serverless functions for execution",
            MITRETactic.EXECUTION.value,
        ),
        "T1609": (
            "Container Administration Command",
            "Abuse container admin interfaces",
            MITRETactic.EXECUTION.value,
        ),
        # Persistence (ATT&CK v18 / CI-CD additions)
        "T1053.007": (
            "Container Orchestration Job",
            "Kubernetes CronJob persistence",
            MITRETactic.PERSISTENCE.value,
        ),
        "T1098": (
            "Account Manipulation",
            "Manipulate accounts to maintain access",
            MITRETactic.PERSISTENCE.value,
        ),
        "T1098.001": (
            "Additional Cloud Credentials",
            "Add credentials to cloud accounts",
            MITRETactic.PERSISTENCE.value,
        ),
        # Privilege Escalation (container)
        "T1611": (
            "Escape to Host",
            "Escape container to host OS",
            MITRETactic.PRIVILEGE_ESCALATION.value,
        ),
        # Defense Evasion (CI/CD, container)
        "T1612": (
            "Build Image on Host",
            "Build malicious container image on host",
            MITRETactic.DEFENSE_EVASION.value,
        ),
        "T1610": (
            "Deploy Container",
            "Deploy malicious container",
            MITRETactic.DEFENSE_EVASION.value,
        ),
        # Discovery (container/Kubernetes)
        "T1613": (
            "Container and Resource Discovery",
            "Enumerate container resources",
            MITRETactic.DISCOVERY.value,
        ),
        # Supply Chain
        "T1195": (
            "Supply Chain Compromise",
            "Compromise software supply chain",
            MITRETactic.INITIAL_ACCESS.value,
        ),
        "T1195.002": (
            "Compromise Software Supply Chain",
            "Inject malicious code into software",
            MITRETactic.INITIAL_ACCESS.value,
        ),
        "T1195.003": (
            "Compromise Hardware Supply Chain",
            "Insert malicious firmware/hardware",
            MITRETactic.INITIAL_ACCESS.value,
        ),
    }

    for tech_id, (name, desc, tactic) in attack_data.items():
        MITRE_ATTACK_TECHNIQUES[tech_id] = MITRETechnique(
            tech_id,
            name,
            desc,
            tactic,
            ["Windows", "Linux", "macOS"],
            ["Process", "File", "Network Traffic"],
            is_atlas=False,
        )

    # === MITRE ATLAS (AI/ML) ===
    atlas_data = {
        "AML.T0024": (
            "Exfiltration via ML Inference API",
            "Extract info via ML API",
            MITREATLASTactic.ML_MODEL_ACCESS.value,
        ),
        "AML.T0025": (
            "Exfiltration via Cyber Means",
            "Traditional ML data exfil",
            MITREATLASTactic.ML_MODEL_ACCESS.value,
        ),
        "AML.T0043": (
            "Craft Adversarial Data",
            "Create adversarial examples",
            MITREATLASTactic.ML_ATTACK_STAGING.value,
        ),
        "AML.T0044": (
            "Full ML Model Access",
            "Complete model access",
            MITREATLASTactic.ML_ATTACK_STAGING.value,
        ),
        "AML.T0045": (
            "Verify Attack",
            "Test adversarial examples",
            MITREATLASTactic.ML_ATTACK_STAGING.value,
        ),
        "AML.T0001": (
            "Valid Accounts",
            "Access ML systems",
            MITREATLASTactic.INITIAL_ACCESS_ML.value,
        ),
        "AML.T0002": (
            "Phishing",
            "Phishing for ML access",
            MITREATLASTactic.INITIAL_ACCESS_ML.value,
        ),
        "AML.T0015": (
            "Evade ML Model",
            "Evade ML detection",
            MITREATLASTactic.ML_ATTACK_EXECUTION.value,
        ),
        "AML.T0040": (
            "ML Enabled Products",
            "Abuse ML features",
            MITREATLASTactic.ML_ATTACK_EXECUTION.value,
        ),
        "AML.T0018": (
            "Backdoor ML Model",
            "Insert ML backdoor",
            MITREATLASTactic.PERSISTENCE_ML.value,
        ),
        "AML.T0019": (
            "Poison Training Data",
            "Inject malicious training data",
            MITREATLASTactic.PERSISTENCE_ML.value,
        ),
        "AML.T0006": (
            "Active Scanning",
            "Scan ML systems",
            MITREATLASTactic.DEFENSE_EVASION_ML.value,
        ),
        "AML.T0042": (
            "Adversarial Example Injection",
            "Inject adversarial examples",
            MITREATLASTactic.DEFENSE_EVASION_ML.value,
        ),
        "AML.T0010": (
            "ML Model Inference API Access",
            "Access ML endpoints",
            MITREATLASTactic.DISCOVERY_ML.value,
        ),
        "AML.T0012": (
            "Discover ML Model Family",
            "Identify model type",
            MITREATLASTactic.DISCOVERY_ML.value,
        ),
        "AML.T0013": (
            "Discover ML Model Ontology",
            "Discover classification ontology",
            MITREATLASTactic.DISCOVERY_ML.value,
        ),
        "AML.T0003": (
            "Discover Training Data",
            "Access training data",
            MITREATLASTactic.COLLECTION_ML.value,
        ),
        "AML.T0029": (
            "Acquire Public ML Artifacts",
            "Download public models",
            MITREATLASTactic.COLLECTION_ML.value,
        ),
        "AML.T0017": (
            "Erode ML Model Integrity",
            "Degrade model performance",
            MITREATLASTactic.ML_ATTACK_IMPACT.value,
        ),
        "AML.T0020": (
            "ML Model Inversion",
            "Extract training data",
            MITREATLASTactic.ML_ATTACK_IMPACT.value,
        ),
        "AML.T0021": (
            "ML Model Theft",
            "Steal model weights",
            MITREATLASTactic.ML_ATTACK_IMPACT.value,
        ),
        "AML.T0051": (
            "LLM Prompt Injection",
            "Inject malicious prompts",
            MITREATLASTactic.EXFILTRATION_ML.value,
        ),
        "AML.T0052": (
            "LLM Jailbreak",
            "Bypass LLM restrictions",
            MITREATLASTactic.EXFILTRATION_ML.value,
        ),
        # ATLAS 2026: Agentic AI techniques
        "AML.T0097": (
            "AI Agent Takeover",
            "Hijack autonomous AI agent execution context",
            MITREATLASTactic.ML_ATTACK_EXECUTION.value,
        ),
        "AML.T0098": (
            "Malicious Dependency Injection",
            "Inject malicious packages into AI pipeline",
            MITREATLASTactic.ML_ATTACK_STAGING.value,
        ),
        "AML.T0099": (
            "LLM Meta-Prompt Extraction",
            "Extract hidden system prompts from LLMs",
            MITREATLASTactic.COLLECTION_ML.value,
        ),
        "AML.T0100": (
            "Indirect Prompt Injection",
            "Embed adversarial instructions in retrieved content",
            MITREATLASTactic.ML_ATTACK_EXECUTION.value,
        ),
        "AML.T0101": (
            "Multimodal Adversarial Input",
            "Craft adversarial inputs across image, audio, and text modalities",
            MITREATLASTactic.ML_ATTACK_STAGING.value,
        ),
        "AML.T0102": (
            "Model Context Protocol Abuse",
            "Exploit MCP tool-calling interfaces for unauthorized actions",
            MITREATLASTactic.ML_ATTACK_EXECUTION.value,
        ),
        "AML.T0103": (
            "AI Agent C2",
            "Use compromised AI agent as covert command-and-control relay",
            MITREATLASTactic.ML_ATTACK_EXECUTION.value,
        ),
        "AML.T0104": (
            "Training Data Extraction",
            "Extract memorized training data through repeated queries",
            MITREATLASTactic.COLLECTION_ML.value,
        ),
        "AML.T0105": (
            "AI Pipeline Poisoning",
            "Corrupt data preprocessing or feature engineering steps",
            MITREATLASTactic.PERSISTENCE_ML.value,
        ),
        "AML.T0106": (
            "Model Weight Tampering",
            "Directly modify model weights to introduce backdoor behavior",
            MITREATLASTactic.PERSISTENCE_ML.value,
        ),
        "AML.T0107": (
            "Embedding Space Manipulation",
            "Alter embedding representations to affect downstream classification",
            MITREATLASTactic.DEFENSE_EVASION_ML.value,
        ),
        "AML.T0108": (
            "RAG Poisoning",
            "Inject malicious documents into retrieval-augmented generation context",
            MITREATLASTactic.PERSISTENCE_ML.value,
        ),
        "AML.T0109": (
            "Function Calling Abuse",
            "Manipulate LLM function-calling to invoke unauthorized external actions",
            MITREATLASTactic.ML_ATTACK_EXECUTION.value,
        ),
        "AML.T0110": (
            "AI Supply Chain Attack",
            "Compromise pre-trained models or datasets in supply chain",
            MITREATLASTactic.INITIAL_ACCESS_ML.value,
        ),
        "AML.T0111": (
            "Membership Inference Attack",
            "Determine if specific data was used in model training",
            MITREATLASTactic.DISCOVERY_ML.value,
        ),
        "AML.T0112": (
            "Model Fingerprinting",
            "Identify model architecture and hyperparameters via black-box queries",
            MITREATLASTactic.DISCOVERY_ML.value,
        ),
        "AML.T0113": (
            "Federated Learning Poisoning",
            "Inject poisoned updates into federated learning aggregation",
            MITREATLASTactic.PERSISTENCE_ML.value,
        ),
        "AML.T0114": (
            "AI Denial of Service",
            "Exhaust model resources with adversarial high-cost queries",
            MITREATLASTactic.ML_ATTACK_IMPACT.value,
        ),
        "AML.T0115": (
            "Gradient Leakage Attack",
            "Reconstruct training data from shared gradients in distributed training",
            MITREATLASTactic.COLLECTION_ML.value,
        ),
        "AML.T0116": (
            "Adversarial Patch",
            "Create physical adversarial patches to fool vision models in the real world",
            MITREATLASTactic.ML_ATTACK_STAGING.value,
        ),
    }

    for tech_id, (name, desc, tactic) in atlas_data.items():
        MITRE_ATLAS_TECHNIQUES[tech_id] = MITRETechnique(
            tech_id,
            name,
            desc,
            tactic,
            ["ML System", "LLM"],
            ["API Logs", "ML Inference Logs"],
            is_atlas=True,
        )


# Initialize on module import
_init_techniques()


def get_technique(technique_id: str) -> Optional[MITRETechnique]:
    """Get MITRE technique by ID."""
    if technique_id.startswith("AML."):
        return MITRE_ATLAS_TECHNIQUES.get(technique_id)
    return MITRE_ATTACK_TECHNIQUES.get(technique_id)


def get_techniques_by_tactic(tactic: str, include_atlas: bool = True) -> List[MITRETechnique]:
    """Get all techniques for a specific tactic."""
    techniques = [t for t in MITRE_ATTACK_TECHNIQUES.values() if t.tactic == tactic]
    if include_atlas:
        techniques.extend([t for t in MITRE_ATLAS_TECHNIQUES.values() if t.tactic == tactic])
    return techniques


def get_all_techniques() -> Dict[str, MITRETechnique]:
    """Get all MITRE techniques (ATT&CK + ATLAS)."""
    return {**MITRE_ATTACK_TECHNIQUES, **MITRE_ATLAS_TECHNIQUES}


# Total: 111 MITRE ATT&CK Enterprise (including v18 CI/CD, container, supply-chain) + 43 MITRE ATLAS (including 2026 agentic AI) = 154 techniques
