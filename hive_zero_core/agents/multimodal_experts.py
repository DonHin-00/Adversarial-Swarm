import torch
import torch.nn as nn
from typing import Optional, Dict
from transformers import ViTModel, AutoModelForCausalLM, AutoTokenizer
from hive_zero_core.agents.base_expert import BaseExpert
from hive_zero_core.scanners.jscrambler.analyzer import JscramblerAnalyzer

class Agent_Vision(BaseExpert):
    """
    Expert 11: Visual Reconnaissance (ViT)
    """
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="Vision", hidden_dim=hidden_dim)
        try:
            self.backbone = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
            self.head = nn.Linear(self.backbone.config.hidden_size, action_dim)
        except:
            self.logger.warning("Failed to load ViT. Using dummy backbone.")
            self.backbone = None
            self.dummy_layer = nn.Linear(3 * 224 * 224, action_dim)

    def analyze_screenshot(self, image_tensor: torch.Tensor) -> torch.Tensor:
        if self.backbone:
            outputs = self.backbone(pixel_values=image_tensor)
            cls_emb = outputs.last_hidden_state[:, 0, :]
            return self.head(cls_emb)
        else:
            return self.dummy_layer(image_tensor.flatten(1))

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() != 4:
            return torch.zeros(x.size(0), self.action_dim, device=x.device)
        return self.analyze_screenshot(x)

class Agent_CodeBreaker(BaseExpert):
    """
    Expert 12: Binary/Source Analysis + Jscrambler Deobfuscation
    """
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="CodeBreaker", hidden_dim=hidden_dim)
        self.encoder = nn.Linear(observation_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, action_dim)
        self.js_analyzer = JscramblerAnalyzer()

    def analyze_snippet(self, code_snippet: str) -> str:
        # 1. Jscrambler Check
        js_report = self.js_analyzer.analyze(code_snippet)
        if js_report["detected_protections"]:
            return f"Detected Jscrambler Protections: {', '.join(js_report['detected_protections'])}"

        # 2. General Vuln Check
        if "buffer" in code_snippet:
            return "Potential Buffer Overflow detected."
        return "No obvious vulnerabilities."

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        feat = torch.relu(self.encoder(x))
        return self.head(feat)
