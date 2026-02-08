import logging
from typing import Any, Dict, List

from mitreattack.stix20 import MitreAttackData


class MitreKnowledgeBase:
    """
    Interface to MITRE ATT&CK STIX Data.
    Maps Techniques to Tools, Malware, and Groups.
    """
    def __init__(self, stix_path: str = "enterprise-attack.json"):
        self.logger = logging.getLogger(__name__)
        try:
            # We assume the json is present or we download it.
            # For prototype, we'll use a mocked lookup if file missing.
            self.mitre_data = MitreAttackData(stix_path)
        except Exception:
            self.logger.warning("MITRE STIX data not found. Using empty KB.")
            self.mitre_data = None

    def get_technique(self, technique_id: str) -> Dict[str, Any]:
        if not self.mitre_data:
            return {"name": "Unknown", "id": technique_id}

        # Look up technique logic
        # mitreattack-python library usage:
        obj = self.mitre_data.get_object_by_attack_id(technique_id, "attack-pattern")
        if obj:
            return {
                "name": obj.name,
                "id": technique_id,
                "description": obj.description,
                "platforms": obj.x_mitre_platforms
            }
        return {}

    def map_service_to_technique(self, service_name: str) -> List[str]:
        """
        Heuristic mapping of Nmap service names to ATT&CK Techniques.
        """
        mapping = {
            "ssh": ["T1021.004"], # SSH
            "http": ["T1190"],    # Exploit Public-Facing App
            "https": ["T1190"],
            "smb": ["T1021.002"], # SMB/Windows Shares
            "rdp": ["T1021.001"], # RDP
            "ftp": ["T1210"],     # Exploit Remote Services
            "mysql": ["T1190"],   # DB Exploitation
        }
        return mapping.get(service_name.lower(), [])
