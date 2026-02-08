import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from typing import Optional, Dict
from hive_zero_core.agents.base_expert import BaseExpert

class Agent_Mimic(BaseExpert):
    """
    Expert 7: Traffic Mimic (Conditional VAE-GAN)
    Generates realistic network traffic shapes.
    Conditioned on Protocol ID to ensure protocol-specific shapes.
    """
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="Mimic", hidden_dim=hidden_dim)

        # Protocol Embedding (Condition)
        self.proto_embedding = nn.Embedding(256, 16)

        # Encoder (VAE)
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim + 16, hidden_dim), # Obs + Proto
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.mu_head = nn.Linear(hidden_dim, hidden_dim)
        self.logvar_head = nn.Linear(hidden_dim, hidden_dim)

        # Decoder (GAN)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + 16 + 16, hidden_dim), # Latent + Noise + Proto
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: Traffic Stats [Batch, Obs]
        # context: Protocol IDs [Batch, 1]? Or embedded?
        if context is None:
             context = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        proto_emb = self.proto_embedding(context.squeeze(-1) if context.dim() > 1 else context)

        # Encoder
        enc_in = torch.cat([x, proto_emb], dim=1)
        enc = self.encoder(enc_in)
        mu = self.mu_head(enc)
        logvar = self.logvar_head(enc)
        z = self._reparameterize(mu, logvar)

        # Decoder
        noise = torch.randn(x.size(0), 16, device=x.device)
        latent_input = torch.cat([z, noise, proto_emb], dim=1)

        out = self.decoder(latent_input)
        return F.softplus(out)

class Agent_Ghost(BaseExpert):
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

class Agent_Stego(BaseExpert):
    """
    Expert 9: DCT Steganography
    Embeds payload in the frequency domain (Discrete Cosine Transform) of the cover image.
    """
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="Stego", hidden_dim=hidden_dim)

        self.payload_encoder = nn.Linear(observation_dim, 64)

    def _dct_2d(self, x):
        return torch.fft.fft2(x).real

    def _idct_2d(self, x):
        return torch.fft.ifft2(x).real

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: Payload [Batch, Obs]
        # context: Cover Image [Batch, 1, 64, 64]

        encoded_msg = torch.tanh(self.payload_encoder(x))

        if context is not None and context.dim() == 4:
             # DCT of full image
             cover = context.squeeze(1) # [Batch, H, W]
             dct_coeffs = self._dct_2d(cover)

             # Embed in mid-frequencies
             B, H, W = dct_coeffs.shape
             flat_dct = dct_coeffs.view(B, -1)

             # Injection
             if flat_dct.size(1) > 100:
                 flat_dct[:, 10:74] += encoded_msg * 0.1

             # IDCT
             stego_img = self._idct_2d(flat_dct.view(B, H, W))
             return stego_img.unsqueeze(1)

        return encoded_msg

class Agent_Cleaner(BaseExpert):
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

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # x: Action History [batch, seq, obs]
        if x.dim() == 2:
            x = x.unsqueeze(1)

        out, _ = self.generator(x)
        action_logits = self.head(out[:, -1, :])

        # Predict State Change
        last_state = x[:, -1, :]
        pred_next_state = self.state_model(torch.cat([last_state, action_logits], dim=1))

        # Check against Initial State
        initial_state = x[:, 0, :]
        diff = F.mse_loss(pred_next_state, initial_state, reduction='none').mean(dim=1, keepdim=True)

        verified = torch.exp(-diff)

        return {"action": action_logits, "verified_score": verified}
