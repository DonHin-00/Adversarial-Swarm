from typing import Optional

import torch
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer

from hive_zero_core.agents.base_expert import BaseExpert
from hive_zero_core.agents.genetic_evolution import GeneticEvolution


def _quantize_to_ids(x: torch.Tensor, vocab_size: int, scale: int = 1000) -> torch.Tensor:
    """Convert a float tensor to token IDs via deterministic quantisation."""
    if x.dtype == torch.float:
        return torch.clamp((torch.abs(x) * scale).long() % vocab_size, 0, vocab_size - 1)
    return torch.clamp(x.long(), 0, vocab_size - 1)


class Agent_Sentinel(BaseExpert):
    """
    Expert 6: The Discriminator (BERT Classifier)

    Classifies payloads as Blocked (0) or Allowed (1) using a pre-trained
    BERT-tiny backbone.  Accepts float tensors (auto-quantised to token IDs),
    integer token IDs, or raw embeddings.  Handles unexpected dimensionality
    gracefully by returning a zero-logit fallback.
    """

    def __init__(self, observation_dim: int, action_dim: int,
                 model_name: str = "prajjwal1/bert-tiny", hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="Sentinel", hidden_dim=hidden_dim)
        self.model_name = model_name

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=2
            )
        except Exception as e:
            self.logger.error(f"Failed to load Sentinel model {model_name}: {e}")
            raise

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor],
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        vocab_size = self.model.config.vocab_size
        max_len = 512

        if x.dim() == 2:
            input_ids = _quantize_to_ids(x, vocab_size, scale=1000)
            if input_ids.size(1) > max_len:
                input_ids = input_ids[:, :max_len]
            outputs = self.model(input_ids=input_ids)

        elif x.dim() == 3:
            target_emb_dim = self.model.config.hidden_size
            if x.size(-1) != target_emb_dim:
                x_flat = x.view(x.size(0) * x.size(1), -1)
                x_adapted = self.ensure_dimension(x_flat, target_emb_dim)
                x = x_adapted.view(x.size(0), x.size(1), target_emb_dim)
            outputs = self.model(inputs_embeds=x)

        else:
            self.logger.warning(f"Unexpected input dims={x.dim()}; returning zero logits")
            return torch.zeros(x.size(0) if x.dim() > 0 else 1, self.action_dim,
                               device=x.device)

        return outputs.logits


class Agent_PayloadGen(BaseExpert):
    """
    Expert 4: Payload Generator (Seq2Seq)

    Generates raw exploit strings from vulnerability context using a
    pre-trained T5-small backbone.  Float observations are quantised to
    token IDs before being fed to the encoder.
    """

    def __init__(self, observation_dim: int, action_dim: int,
                 model_name: str = "t5-small", hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="PayloadGen", hidden_dim=hidden_dim)
        self.model_name = model_name

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        except Exception as e:
            self.logger.error(f"Failed to load PayloadGen model {model_name}: {e}")
            raise

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor],
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        max_len = 128
        vocab_size = self.model.config.vocab_size

        input_ids = _quantize_to_ids(x, vocab_size, scale=100)
        outputs = self.model.generate(input_ids, max_length=max_len, do_sample=True)

        return self.ensure_dimension(outputs.float(), self.action_dim)


class Agent_Mutator(BaseExpert):
    """
    Expert 5: Adversarial Payload Optimiser with Genetic Evolution

    Iteratively obfuscates a generated payload to maximise the probability
    of evading the Sentinel classifier.  Uses gradient-based embedding-space
    search with Adam (upgraded from SGD) and configurable optimisation steps.
    Log-softmax is used for numerically stable loss computation.

    Enhanced with genetic evolution capabilities:
    - Polymorphic engine for signature evasion
    - Natural selection for payload validation
    - Generation tracking for evolution monitoring
    """

    def __init__(self, observation_dim: int, action_dim: int,
                 sentinel_expert: BaseExpert, generator_expert: BaseExpert,
                 hidden_dim: int = 64, k_steps: int = 5, lr: float = 0.05,
                 enable_evolution: bool = True):
        super().__init__(observation_dim, action_dim, name="Mutator", hidden_dim=hidden_dim)

        self.sentinel = sentinel_expert
        self.generator = generator_expert
        self.k_steps = k_steps
        self.lr = lr

        # Genetic Evolution Engine
        self.enable_evolution = enable_evolution
        if enable_evolution:
            self.evolution_engine = GeneticEvolution(max_generations=100)
        else:
            self.evolution_engine = None

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor],
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Inference-time adversarial search loop."""
        # 1. Initial generation (no gradients needed)
        with torch.no_grad():
            vocab_size = self.generator.model.config.vocab_size
            input_ids = _quantize_to_ids(x, vocab_size, scale=100)
            initial_token_ids = self.generator.model.generate(
                input_ids, max_length=64, do_sample=True
            )

        # 2. Convert to embeddings for gradient-based optimisation
        embed_layer = self.sentinel.model.get_input_embeddings()
        sentinel_vocab_size = self.sentinel.model.config.vocab_size

        initial_token_ids = torch.clamp(
            initial_token_ids.long(), 0, sentinel_vocab_size - 1
        )
        if initial_token_ids.shape[-1] > 512:
            initial_token_ids = initial_token_ids[:, :512]

        current_embeddings = embed_layer(initial_token_ids).clone().detach()
        current_embeddings.requires_grad_(True)

        # Adam for faster convergence (upgraded from SGD)
        optimizer = optim.Adam([current_embeddings], lr=self.lr)

        best_embeddings = current_embeddings.clone().detach()
        best_score = -1.0

        # 3. Optimisation loop
        for _ in range(self.k_steps):
            optimizer.zero_grad()

            outputs = self.sentinel.model(inputs_embeds=current_embeddings)
            logits = outputs.logits

            # Numerically stable: use log_softmax directly
            log_probs = F.log_softmax(logits, dim=-1)
            score = torch.exp(log_probs[:, 1]).mean()

            if score.item() > best_score:
                best_score = score.item()
                best_embeddings = current_embeddings.clone().detach()

            # Maximise P(Allowed) via negative log-likelihood
            loss = -log_probs[:, 1].mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([current_embeddings], max_norm=1.0)
            optimizer.step()

        # 4. Flatten and adapt to action_dim
        flat = best_embeddings.view(x.size(0), -1)
        return self.ensure_dimension(flat, self.action_dim)

    def evolve_payload_text(self, payload_text: str, max_attempts: int = 10) -> tuple:
        """
        Apply genetic evolution to a text payload for signature evasion.

        Args:
            payload_text: Original payload string
            max_attempts: Maximum mutation attempts

        Returns:
            Tuple of (mutated_payload, gene_seed, success)
        """
        if not self.enable_evolution or self.evolution_engine is None:
            self.logger.warning("Genetic evolution not enabled")
            return payload_text, 0, False

        mutated, gene_seed, success = self.evolution_engine.evolve_payload(
            payload_text, max_attempts=max_attempts
        )

        if success:
            self.logger.info(f"Payload evolved successfully (gen {self.evolution_engine.tracker.current_generation})")
        else:
            self.logger.warning("Payload evolution failed, using original")

        return mutated, gene_seed, success

    def evolve_code(self, source_code: str, max_attempts: int = 10) -> tuple:
        """
        Apply genetic evolution to Python source code for polymorphic capabilities.

        Args:
            source_code: Original Python source code
            max_attempts: Maximum mutation attempts

        Returns:
            Tuple of (mutated_code, gene_seed, success)
        """
        if not self.enable_evolution or self.evolution_engine is None:
            self.logger.warning("Genetic evolution not enabled")
            return source_code, 0, False

        mutated, gene_seed, success = self.evolution_engine.evolve_code(
            source_code, max_attempts=max_attempts
        )

        if success:
            self.logger.info(f"Code evolved successfully (gen {self.evolution_engine.tracker.current_generation})")
        else:
            self.logger.warning("Code evolution failed, using original")

        return mutated, gene_seed, success

    def get_evolution_stats(self) -> dict:
        """
        Get statistics about genetic evolution performance.

        Returns:
            Dictionary with evolution statistics
        """
        if not self.enable_evolution or self.evolution_engine is None:
            return {'enabled': False}

        stats = self.evolution_engine.get_stats()
        stats['enabled'] = True
        return stats
