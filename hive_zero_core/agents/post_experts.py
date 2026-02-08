import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from typing import Optional, Dict
from hive_zero_core.agents.base_expert import BaseExpert

class MimicAgent(BaseExpert):
    """
    Expert 7: Traffic Mimic (Conditional VAE-GAN)
    Generates realistic network traffic shapes using a VAE-GAN architecture.
    """
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="Mimic", hidden_dim=hidden_dim)

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

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if context is None:
            context = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        proto_emb = self.proto_embedding(context.squeeze(-1) if context.dim() > 1 else context)

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
    Expert 8: Kernel Metadata Analyzer
    Analyzes simulated kernel structures (inodes, syscalls) for hiding spots.
    """
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="Ghost", hidden_dim=hidden_dim)

        # Kernel Feature Extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Risk Assessment Head (Probability of Syscall Hook Detection)
        self.risk_head = nn.Linear(hidden_dim, 1)
        # Suitability Head
        self.score_head = nn.Linear(hidden_dim, 1)

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
    Expert 10: Formal Logic Verifier
    Generates cleanup scripts and verifies state transition.
    """
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="Cleaner", hidden_dim=hidden_dim)

        # History needs [batch, seq, obs]
        self.generator = nn.LSTM(observation_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, action_dim)

        # State Predictor
        self.state_model = nn.Sequential(
            nn.Linear(observation_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, observation_dim)
        )

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
