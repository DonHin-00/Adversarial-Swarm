import re
from typing import Dict, Any

class NLCommandProcessor:
    """
    Natural Language Interface.
    Translates English commands into Strategic Goals and Constraints.
    """
    def __init__(self):
        # Rules-based parsing for prototype (Mocking LLM)
        self.patterns = {
            r"pivot to (.*) using (.*)": self._handle_pivot,
            r"scan subnet (.*)": self._handle_scan,
            r"deploy persistence on (.*)": self._handle_persist
        }

    def process_command(self, text: str) -> Dict[str, Any]:
        text = text.lower()
        for pattern, handler in self.patterns.items():
            match = re.search(pattern, text)
            if match:
                return handler(match)
        return {"error": "Command not understood"}

    def _handle_pivot(self, match):
        return {
            "strategy": 1, # Infiltrate
            "focus_ip": match.group(1),
            "preferred_technique": match.group(2)
        }

    def _handle_scan(self, match):
        return {
            "strategy": 0, # Recon
            "target_range": match.group(1)
        }

    def _handle_persist(self, match):
        return {
            "strategy": 2, # Persist
            "target_ip": match.group(1)
        }
