from typing import Tuple

import torch
from torch import nn


class SafetyMonitor(nn.Module):
    """
    The 'Kill Switch'. Monitors Alert Levels and Action Validity.
    Overrides Agent actions if constraints are violated.
    """
    def __init__(self, alert_threshold: float = 0.8):
        super().__init__()
        self.alert_threshold = alert_threshold
        # Simple learnable component to predict future risk?
        # Or just deterministic rules.
        # Let's add a small risk predictor.
        self.risk_predictor = nn.Linear(64, 1) # From state

    def check_safety(self, state: torch.Tensor, action_vector: torch.Tensor, current_alert_level: float) -> Tuple[bool, str]:
        """
        Returns (is_safe, reason)
        """
        # 1. Hard Threshold Check
        if current_alert_level > self.alert_threshold:
            return False, "Alert Threshold Exceeded"

        # 2. Predicted Risk Check
        with torch.no_grad():
            risk = torch.sigmoid(self.risk_predictor(state))

        if risk.item() > 0.95:  # noqa: PLR2004
            return False, "High Risk Predicted"

        # 3. Action Sanity Check (e.g. Norm)
        if torch.norm(action_vector) > 10.0:  # noqa: PLR2004
             return False, "Action Magnitude Unsafe"

        return True, "Safe"

    def forward(self, state):
        # Used for training the risk predictor
        return torch.sigmoid(self.risk_predictor(state))
