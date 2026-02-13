"""
Evolving Threat Intelligence Database (ThreatIntelDB)

A persistent, co-evolutionary memory store that tracks attack signatures,
defence patterns, and engagement outcomes across training iterations.
Designed to simulate a continuously-adapting threat analyst: it records
every payload that successfully evaded (or was caught by) the blue-team
stack, distills the embeddings into a compact prototype bank, and
exposes query methods that both red- and blue-team agents can use to
inform their next decisions.

Key Concepts
------------
*Signature*
    A compact embedding vector that represents a family of similar
    attack or defence patterns.  Signatures are grouped into two banks:

    1. **attack_bank** — embeddings of payloads that *succeeded* in
       evading detection.  The red team queries this bank to seed
       new payloads (imitation of past success).

    2. **defense_bank** — embeddings of payloads that *failed*
       (were caught).  The blue team queries this bank to recognise
       recurring attack patterns.

*EMA Consolidation*
    Both banks are updated via exponential moving average (EMA) so that
    recent experience has higher weight, but historical patterns are
    never fully forgotten — just like a veteran threat analyst who
    remembers past campaigns while adapting to new ones.

*Novelty Signal*
    When a new payload is submitted, its minimum distance to the
    closest existing signature is returned as a *novelty score*.
    High-novelty payloads are preferentially stored, ensuring the
    database doesn't collapse into a narrow mode.

*Generational Counter*
    A monotonically-increasing generation counter tracks how many
    training cycles the database has absorbed.  Downstream agents
    can use this to scale exploration noise or learning rates.

Threading Safety
----------------
All mutations are designed to run under ``torch.no_grad()`` on a
single training loop; no locks are required.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any


class ThreatIntelDB(nn.Module):
    """
    Persistent co-evolutionary memory for red/blue adversarial training.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of payload/detection embeddings.
    bank_size : int
        Maximum number of prototype signatures per bank.
    ema_decay : float
        Exponential moving average decay for signature updates.
    novelty_threshold : float
        Minimum cosine distance for a payload to be considered novel
        enough to enter the bank (prevents mode collapse).
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        bank_size: int = 256,
        ema_decay: float = 0.995,
        novelty_threshold: float = 0.1,
        novelty_warmup_gens: int = 50,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.bank_size = bank_size
        self.ema_decay = ema_decay
        self.novelty_threshold = novelty_threshold
        # Over the first `novelty_warmup_gens` generations, the effective
        # novelty threshold linearly decays from 1.0 (accept everything)
        # to `novelty_threshold`.  This ensures the bank fills quickly
        # in early training, then becomes more selective as patterns emerge.
        self.novelty_warmup_gens = novelty_warmup_gens

        # Attack bank: successful evasions the red team can draw from
        self.register_buffer(
            "attack_bank",
            torch.randn(bank_size, embedding_dim) * 0.01,
        )
        self.register_buffer("attack_count", torch.zeros(bank_size))
        self.register_buffer("_attack_ptr", torch.tensor(0, dtype=torch.long))

        # Defence bank: caught payloads the blue team learns from
        self.register_buffer(
            "defense_bank",
            torch.randn(bank_size, embedding_dim) * 0.01,
        )
        self.register_buffer("defense_count", torch.zeros(bank_size))
        self.register_buffer("_defense_ptr", torch.tensor(0, dtype=torch.long))

        # Generational counter
        self.register_buffer("generation", torch.tensor(0, dtype=torch.long))

        # Per-generation aggregate statistics (ring buffer of last 100)
        self.register_buffer("evasion_rate_history", torch.zeros(100))
        self.register_buffer("_history_ptr", torch.tensor(0, dtype=torch.long))

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    @torch.no_grad()
    def record_attack_success(self, embeddings: torch.Tensor):
        """
        Record payload embeddings that *successfully evaded* all blue detectors.

        Args:
            embeddings: [N, embedding_dim] — successful payload embeddings.
        """
        self._ingest(embeddings, self.attack_bank, self.attack_count, self._attack_ptr)

    @torch.no_grad()
    def record_attack_failure(self, embeddings: torch.Tensor):
        """
        Record payload embeddings that were *caught* by at least one detector.

        Args:
            embeddings: [N, embedding_dim] — failed payload embeddings.
        """
        self._ingest(embeddings, self.defense_bank, self.defense_count, self._defense_ptr)

    @torch.no_grad()
    def step_generation(self, evasion_rate: float):
        """
        Advance the generational counter and log the evasion rate.

        Args:
            evasion_rate: Fraction of payloads that evaded detection this gen.
        """
        idx = self._history_ptr.item() % 100
        self.evasion_rate_history[idx] = evasion_rate
        self._history_ptr += 1
        self.generation += 1

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query_attack_bank(
        self, query: torch.Tensor, top_k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the *top_k* most similar signatures from the attack bank.

        Args:
            query: [B, embedding_dim] — query embeddings.
            top_k: Number of neighbours to return.

        Returns:
            (signatures, similarities): ([B, top_k, D], [B, top_k])
        """
        return self._query(query, self.attack_bank, top_k)

    def query_defense_bank(
        self, query: torch.Tensor, top_k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the *top_k* most similar signatures from the defence bank.
        """
        return self._query(query, self.defense_bank, top_k)

    def novelty_score(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute the novelty of each embedding relative to *both* banks.

        Returns:
            [N] tensor of novelty scores (higher = more novel).
        """
        all_sigs = torch.cat([self.attack_bank, self.defense_bank], dim=0)
        emb_norm = F.normalize(embeddings, dim=-1)
        sig_norm = F.normalize(all_sigs, dim=-1)

        cos_sim = torch.matmul(emb_norm, sig_norm.t())  # [N, 2*bank]
        max_sim = cos_sim.max(dim=-1).values  # [N]

        # Novelty = 1 − max_similarity
        return 1.0 - max_sim

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics for diagnostics."""
        gen = self.generation.item()
        n_history = min(gen, 100)
        if n_history > 0:
            rates = self.evasion_rate_history[:n_history]
            avg_evasion = rates.mean().item()
            trend = (rates[-1] - rates[0]).item() if n_history > 1 else 0.0
        else:
            avg_evasion = 0.0
            trend = 0.0

        return {
            "generation": gen,
            "attack_bank_fill": int((self.attack_count > 0).sum().item()),
            "defense_bank_fill": int((self.defense_count > 0).sum().item()),
            "avg_evasion_rate": avg_evasion,
            "evasion_trend": trend,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _effective_novelty_threshold(self) -> float:
        """Temperature-scaled novelty: starts permissive, tightens over time."""
        gen = self.generation.item()
        if gen >= self.novelty_warmup_gens:
            return self.novelty_threshold
        # Linear interpolation: 1.0 → novelty_threshold over warmup period
        alpha = gen / max(self.novelty_warmup_gens, 1)
        return 1.0 * (1 - alpha) + self.novelty_threshold * alpha

    def _ingest(
        self, embeddings: torch.Tensor, bank: torch.Tensor, count: torch.Tensor, ptr: torch.Tensor
    ):
        """Insert or EMA-update signatures in the given bank.

        Normalises the bank once up-front (and re-normalises only the
        affected rows after an update) to avoid O(N * bank_size * D)
        repeated full-bank normalisation.

        Uses a temperature-scaled novelty threshold: during early training
        (first ``novelty_warmup_gens`` generations) the threshold is relaxed
        to fill the bank quickly, then tightens to the configured value.
        """
        bank_norm = F.normalize(bank, dim=-1)  # Normalise once
        threshold = self._effective_novelty_threshold()

        for i in range(embeddings.size(0)):
            emb = embeddings[i]

            # Check novelty against this bank
            emb_norm = F.normalize(emb.unsqueeze(0), dim=-1)
            cos_sim = torch.matmul(emb_norm, bank_norm.t()).squeeze(0)
            max_sim, max_idx = cos_sim.max(dim=0)

            if max_sim.item() > (1.0 - threshold):
                # Close match exists → EMA update
                bank[max_idx] = self.ema_decay * bank[max_idx] + (1 - self.ema_decay) * emb
                count[max_idx] += 1
                # Re-normalise only the updated row
                bank_norm[max_idx] = F.normalize(bank[max_idx].unsqueeze(0), dim=-1).squeeze(0)
            else:
                # Novel entry → overwrite at pointer position
                idx = ptr.item() % bank.size(0)
                bank[idx] = emb
                count[idx] = 1
                ptr += 1
                # Re-normalise only the new row
                bank_norm[idx] = F.normalize(bank[idx].unsqueeze(0), dim=-1).squeeze(0)

    def _query(
        self, query: torch.Tensor, bank: torch.Tensor, top_k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cosine-similarity kNN against a bank."""
        q_norm = F.normalize(query, dim=-1)  # [B, D]
        b_norm = F.normalize(bank, dim=-1)  # [S, D]

        sim = torch.matmul(q_norm, b_norm.t())  # [B, S]
        effective_k = max(1, min(top_k, bank.size(0)))
        top_sim, top_idx = torch.topk(sim, k=effective_k, dim=-1)

        # Gather signatures
        top_sigs = bank[top_idx]  # [B, K, D]

        return top_sigs, top_sim
