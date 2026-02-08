import os  # noqa: I001
import torch
import torch.nn as nn  # noqa: PLR0402
import torch.nn.functional as F  # noqa: N812
import torch.fft
from typing import Optional, Dict, List, Union  # noqa: F401
from hive_zero_core.agents.base_expert import BaseExpert


class Agent_Mimic(BaseExpert):  # noqa: N801
    """
    Expert 7: Traffic Mimicry (C-VAE-GAN)
class MimicAgent(BaseExpert):
    """
    Expert 7: Traffic Mimic (Conditional VAE-GAN)
    Generates realistic network traffic shapes using a VAE-GAN architecture.
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="Mimic", hidden_dim=hidden_dim)
        self.proto_embedding = nn.Embedding(256, 16)
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim + 16, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.mu_head = nn.Linear(hidden_dim, hidden_dim)
        self.logvar_head = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + 16 + 16, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),

        self.proto_embedding = nn.Embedding(256, 16)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim + 16, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.mu_head = nn.Linear(hidden_dim, hidden_dim)
        self.logvar_head = nn.Linear(hidden_dim, hidden_dim)

        # Decoder / Generator
        self.generator = nn.Sequential(
            nn.Linear(hidden_dim + 16, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(action_dim + 16, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _forward_impl(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if context is None:
            context = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Assume context is protocol ID
        if context.dim() > 1:
            context = context.squeeze(-1)

        proto_emb = self.proto_embedding(context.long())
        enc_in = torch.cat([x, proto_emb], dim=1)
        enc = self.encoder(enc_in)
        z = self._reparameterize(self.mu_head(enc), self.logvar_head(enc))
        noise = torch.randn(x.size(0), 16, device=x.device)
        return F.softplus(self.decoder(torch.cat([z, noise, proto_emb], dim=1)))


class Agent_Ghost(BaseExpert):  # noqa: N801
        # 1. Encode to Latent
        enc_in = torch.cat([x, proto_emb], dim=1)
        enc = self.encoder(enc_in)
        mu, logvar = self.mu_head(enc), self.logvar_head(enc)
        z = self._reparameterize(mu, logvar)

        # 2. Generate Traffic Shape
        gen_in = torch.cat([z, proto_emb], dim=1)
        generated_traffic = self.generator(gen_in)

        return generated_traffic

class GhostAgent(BaseExpert):
    """
    Expert 8: Persistence / File System Ghost
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="Ghost", hidden_dim=hidden_dim)
        self.feature_extractor = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.score_head = nn.Linear(hidden_dim, 1)

    def analyze_path(self, path: str) -> Dict[str, float]:
        score = 0.0
        try:
            stats = os.stat(path)
            mode = stats.st_mode
            if mode & 0o002:
                score += 0.8  # noqa: E701
            if mode & 0o020:
                score += 0.4  # noqa: E701
            if "tmp" in path:
                score += 0.5  # noqa: E701
            if "log" in path:
                score += 0.3  # noqa: E701
        except Exception:
            score = -1.0
        return {"suitability": score}

    def _forward_impl(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if x.dim() == 2:  # noqa: PLR2004
            feat = self.feature_extractor(x)
            return torch.sigmoid(self.score_head(feat)) * (1.0 - torch.sigmoid(self.risk_head(feat)))
        return torch.zeros(x.size(0), 1, device=x.device)


class Agent_Stego(BaseExpert):  # noqa: N801
    """
    Expert 9: Steganography
    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: Metadata
        feat = self.feature_extractor(x)
        score = torch.sigmoid(self.score_head(feat))
        risk = torch.sigmoid(self.risk_head(feat))

        # Net Utility = Suitability * (1 - Risk)
        return score * (1.0 - risk)

class StegoAgent(BaseExpert):
    """
    Expert 9: Frequency-Domain Steganography
    Embeds payloads into the DCT coefficients of cover images.
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="Stego", hidden_dim=hidden_dim)
        self.payload_encoder = nn.Linear(observation_dim, 64)

    def _dct_2d(self, x):
        return torch.fft.fft2(x).real

    def _idct_2d(self, x):
        return torch.fft.ifft2(x).real

    def _forward_impl(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        encoded_msg = torch.tanh(self.payload_encoder(x))
        if context is not None and context.dim() == 4:  # noqa: PLR2004
            cover = context.squeeze(1)
            dct_coeffs = self._dct_2d(cover)
            flat_dct = dct_coeffs.view(dct_coeffs.shape[0], -1)
            if flat_dct.size(1) > 100:  # noqa: PLR2004
                flat_dct[:, 10:74] += encoded_msg * 0.1
            stego_img = self._idct_2d(flat_dct.view(cover.shape))
            return stego_img.unsqueeze(1)
        return encoded_msg


class Agent_Cleaner(BaseExpert):  # noqa: N801
    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: Payload [Batch, Obs]
        # context: Cover Image [Batch, 1, H, W]
        encoded_msg = torch.tanh(self.payload_encoder(x))

        if context is not None and context.dim() == 4:
             cover = context.squeeze(1)
             dct_coeffs = torch.fft.rfft2(cover, norm='ortho')

             # Injection into mid-frequency components
             B, H, W_half = dct_coeffs.shape
             flat_dct = dct_coeffs.view(B, -1)

             if flat_dct.size(1) > 64:
                 # Real/Imaginary injection
                 flat_dct[:, 5:69] += encoded_msg * 0.05 + 1j * (encoded_msg * 0.05)

             stego_img = torch.fft.irfft2(flat_dct.view(B, H, W_half), s=cover.shape[-2:], norm='ortho')
             return stego_img.unsqueeze(1)

        return encoded_msg

class CleanerAgent(BaseExpert):
    """
    Expert 10: Trace Cleanup
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="Cleaner", hidden_dim=hidden_dim)
        self.generator = nn.LSTM(observation_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, action_dim)
        # Added missing model definition from dangling code
        self.state_model = nn.Sequential(
            nn.Linear(observation_dim + action_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, observation_dim)
        )

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

    def _forward_impl(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if x.dim() == 2:  # noqa: PLR2004
    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: Action History [batch, seq, obs]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.generator(x)
        action_logits = self.head(out[:, -1, :])

        # Verification Score (can be used for internal logging or aux loss)
        with torch.no_grad():
            last_state = x[:, -1, :]
            pred_next_state = self.state_model(torch.cat([last_state, action_logits], dim=1))
            initial_state = x[:, 0, :]
            diff = F.mse_loss(pred_next_state, initial_state, reduction='none').mean(dim=1, keepdim=True)
            self.last_verified_score = torch.exp(-diff)

        return action_logits
