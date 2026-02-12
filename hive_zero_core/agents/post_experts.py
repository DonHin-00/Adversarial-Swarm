import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from typing import Optional, Dict, List
from hive_zero_core.agents.base_expert import BaseExpert
import os

class Agent_Mimic(BaseExpert):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="Mimic", hidden_dim=hidden_dim)
        self.proto_embedding = nn.Embedding(256, 16)
        self.encoder = nn.Sequential(nn.Linear(observation_dim + 16, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.mu_head = nn.Linear(hidden_dim, hidden_dim)
        self.logvar_head = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.Sequential(nn.Linear(hidden_dim + 16 + 16, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_dim))

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if context is None: context = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        proto_emb = self.proto_embedding(context.squeeze(-1) if context.dim() > 1 else context)
        enc_in = torch.cat([x, proto_emb], dim=1)
        enc = self.encoder(enc_in)
        z = self._reparameterize(self.mu_head(enc), self.logvar_head(enc))
        noise = torch.randn(x.size(0), 16, device=x.device)
        return F.softplus(self.decoder(torch.cat([z, noise, proto_emb], dim=1)))

class Agent_Ghost(BaseExpert):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="Ghost", hidden_dim=hidden_dim)
        self.feature_extractor = nn.Sequential(nn.Linear(observation_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.score_head = nn.Linear(hidden_dim, 1)

    def analyze_path(self, path: str) -> Dict[str, float]:
        score = 0.0
        try:
            stats = os.stat(path)
            mode = stats.st_mode
            if mode & 0o002: score += 0.8
            if mode & 0o020: score += 0.4
            if "tmp" in path: score += 0.5
            if "log" in path: score += 0.3
        except Exception:
            score = -1.0
        return {"suitability": score}

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() == 2:
            feat = self.feature_extractor(x)
            return torch.sigmoid(self.score_head(feat)) * (1.0 - torch.sigmoid(self.risk_head(feat)))
        return torch.zeros(x.size(0), 1, device=x.device)

class Agent_Stego(BaseExpert):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="Stego", hidden_dim=hidden_dim)
        self.payload_encoder = nn.Linear(observation_dim, 64)

    def _dct_2d(self, x): return torch.fft.fft2(x).real
    def _idct_2d(self, x): return torch.fft.ifft2(x).real

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        encoded_msg = torch.tanh(self.payload_encoder(x))
        if context is not None and context.dim() == 4:
             cover = context.squeeze(1)
             dct_coeffs = self._dct_2d(cover)
             flat_dct = dct_coeffs.view(dct_coeffs.shape[0], -1)
             if flat_dct.size(1) > 100:
                 flat_dct[:, 10:74] += encoded_msg * 0.1
             stego_img = self._idct_2d(flat_dct.view(cover.shape))
             return stego_img.unsqueeze(1)
        return encoded_msg

class Agent_Cleaner(BaseExpert):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="Cleaner", hidden_dim=hidden_dim)
        self.generator = nn.LSTM(observation_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, action_dim)

    def generate_cleanup_script(self, action_history: List[str]) -> str:
        script = []
        for action in reversed(action_history):
            if "touch" in action:
                filename = action.split()[-1]
                script.append(f"rm {filename}")
            elif "mkdir" in action:
                dirname = action.split()[-1]
                script.append(f"rmdir {dirname}")
            elif "curl" in action or "wget" in action:
                script.append("# Check for downloaded artifacts")

        script.append("history -c")
        return "\n".join(script)

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.generator(x)
        action_logits = self.head(out[:, -1, :])
        return action_logits
