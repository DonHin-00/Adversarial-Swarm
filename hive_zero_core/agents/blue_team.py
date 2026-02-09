"""
Blue Team Detection Stack — WAF, EDR, SIEM, IDS

Provides a suite of adversarial detection agents that serve as the defensive
counterpart to the red-team swarm.  During training, these agents act as
increasingly difficult adversaries that the red-team payload pipeline must
learn to evade, creating an arms-race co-evolution loop.

Architecture
------------
Agent_WAF          Adversarial Web Application Firewall — classifies HTTP
                   payloads as malicious/benign using a learned rule stack
                   and signature memory bank.

Agent_EDR          Endpoint Detection & Response — analyses process-level
                   behavioural telemetry for anomalous sequences.

Agent_SIEM         Security Information & Event Management — correlates
                   multi-source log events and scores aggregate threat.

Agent_IDS          Intrusion Detection System — deep-packet inspection
                   using convolutional feature extraction over raw payload
                   byte sequences.

All blue-team agents share the BaseExpert interface so they can be gated
and orchestrated by the HiveMind like any other expert.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from hive_zero_core.agents.base_expert import BaseExpert


# ---------------------------------------------------------------------------
# Agent_WAF — Adversarial Web Application Firewall
# ---------------------------------------------------------------------------

class Agent_WAF(BaseExpert):
    """
    Adversarial WAF with a learnable signature memory bank.

    Maintains an internal *signature bank* — a set of learned prototype
    vectors representing known-malicious patterns.  Incoming payloads are
    compared against the bank via scaled dot-product attention; the
    maximum similarity score determines the detection confidence.

    The bank is updated every training step via an exponential moving
    average (EMA) of the most-suspicious payload embeddings, ensuring the
    WAF continuously adapts to novel attack patterns.
    """

    def __init__(self, observation_dim: int, action_dim: int = 2,
                 hidden_dim: int = 128, num_signatures: int = 64,
                 ema_decay: float = 0.99):
        super().__init__(observation_dim, action_dim, name="WAF", hidden_dim=hidden_dim)

        self.num_signatures = num_signatures
        self.ema_decay = ema_decay

        # Feature extractor
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Learnable signature bank: [num_signatures, hidden_dim]
        self.register_buffer(
            "signature_bank",
            torch.randn(num_signatures, hidden_dim) * 0.02,
        )

        # Classification head: maps similarity vector → [Blocked, Allowed]
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + num_signatures, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Rule-stack: lightweight per-rule scoring
        self.rule_weights = nn.Parameter(torch.ones(num_signatures) / num_signatures)

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor],
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Payload embedding [batch, observation_dim].

        Returns:
            Logits [batch, 2] — [P(Blocked), P(Allowed)].
        """
        features = self.encoder(x)  # [B, H]

        # Scaled dot-product similarity against signature bank
        # [B, H] × [H, S] → [B, S]
        sim = torch.matmul(features, self.signature_bank.t())
        sim = sim / (self.signature_bank.size(1) ** 0.5)

        # Weighted rule-stack scoring
        weighted_sim = sim * F.softmax(self.rule_weights, dim=0)

        # Concatenate features + similarity for classification
        combined = torch.cat([features, weighted_sim], dim=-1)
        logits = self.classifier(combined)

        return logits

    @torch.no_grad()
    def update_signatures(self, payload_embeddings: torch.Tensor):
        """
        EMA update of the signature bank with recent suspicious embeddings.

        Args:
            payload_embeddings: [N, hidden_dim] — encoded payloads that
                                triggered high-confidence detections.
        """
        if payload_embeddings.size(0) == 0:
            return

        # Pick top-N closest signatures to update
        features = payload_embeddings[:self.num_signatures]
        n = min(features.size(0), self.num_signatures)

        self.signature_bank[:n] = (
            self.ema_decay * self.signature_bank[:n]
            + (1 - self.ema_decay) * features[:n]
        )


# ---------------------------------------------------------------------------
# Agent_EDR — Endpoint Detection & Response
# ---------------------------------------------------------------------------

class Agent_EDR(BaseExpert):
    """
    Behavioural EDR using a Transformer encoder over process-level telemetry.

    Models temporal sequences of system-call / process events to detect
    anomalous execution patterns.  Uses a small Transformer with learned
    positional embeddings and a binary anomaly head.
    """

    def __init__(self, observation_dim: int, action_dim: int = 2,
                 hidden_dim: int = 128, nhead: int = 4, num_layers: int = 2):
        super().__init__(observation_dim, action_dim, name="EDR", hidden_dim=hidden_dim)

        self.input_proj = nn.Linear(observation_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 512, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 4,
            dropout=0.1, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

        # Anomaly head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor],
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, D] → [B, 1, D]

        batch, seq_len, _ = x.shape
        h = self.input_proj(x) + self.pos_embed[:, :seq_len, :]
        h = self.encoder(h)
        h = self.norm(h)

        # Pool over sequence
        pooled = h.mean(dim=1)  # [B, H]
        return self.head(pooled)


# ---------------------------------------------------------------------------
# Agent_SIEM — Security Information & Event Management
# ---------------------------------------------------------------------------

class Agent_SIEM(BaseExpert):
    """
    Multi-source log correlator and aggregate threat scorer.

    Accepts a global-state observation (the mean-pooled graph embedding
    from LogEncoder) and computes a threat-level score by cross-attending
    over a bank of *learned alert prototypes* — abstract representations
    of known multi-stage attack patterns (e.g. scan→exploit→exfil).
    """

    def __init__(self, observation_dim: int, action_dim: int = 2,
                 hidden_dim: int = 128, num_alert_prototypes: int = 16,
                 nhead: int = 4):
        super().__init__(observation_dim, action_dim, name="SIEM", hidden_dim=hidden_dim)

        self.num_prototypes = num_alert_prototypes

        # Learned alert prototypes
        self.alert_prototypes = nn.Parameter(
            torch.randn(1, num_alert_prototypes, hidden_dim) * 0.02
        )

        self.obs_proj = nn.Linear(observation_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=nhead, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

        self.threat_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor],
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch = x.size(0)

        # Project observation as query
        query = self.obs_proj(x).unsqueeze(1)  # [B, 1, H]

        # Alert prototypes as keys/values (shared across batch)
        kv = self.alert_prototypes.expand(batch, -1, -1)  # [B, P, H]

        attn_out, _ = self.cross_attn(query, kv, kv)  # [B, 1, H]
        attn_out = self.norm(attn_out + query)  # Residual
        pooled = attn_out.squeeze(1)  # [B, H]

        return self.threat_head(pooled)


# ---------------------------------------------------------------------------
# Agent_IDS — Intrusion Detection System
# ---------------------------------------------------------------------------

class Agent_IDS(BaseExpert):
    """
    Deep-Packet Inspection IDS using 1-D convolutions.

    Treats the observation vector as a pseudo byte-sequence and applies
    multi-scale 1-D convolutions (kernel sizes 3, 5, 7) to extract
    n-gram-like features, then classifies via a pooled representation.
    """

    def __init__(self, observation_dim: int, action_dim: int = 2,
                 hidden_dim: int = 128, num_filters: int = 64):
        super().__init__(observation_dim, action_dim, name="IDS", hidden_dim=hidden_dim)

        # Multi-scale 1-D convolutions
        self.conv3 = nn.Conv1d(1, num_filters, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(1, num_filters, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(1, num_filters, kernel_size=7, padding=3)

        self.norm = nn.LayerNorm(num_filters * 3)

        self.classifier = nn.Sequential(
            nn.Linear(num_filters * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor],
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, D] → [B, 1, D] for Conv1d
        x_1d = x.unsqueeze(1)

        c3 = F.gelu(self.conv3(x_1d))  # [B, F, D]
        c5 = F.gelu(self.conv5(x_1d))
        c7 = F.gelu(self.conv7(x_1d))

        # Global max-pool over the sequence dimension
        p3 = c3.max(dim=-1).values  # [B, F]
        p5 = c5.max(dim=-1).values
        p7 = c7.max(dim=-1).values

        combined = self.norm(torch.cat([p3, p5, p7], dim=-1))  # [B, 3F]
        return self.classifier(combined)
