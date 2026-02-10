from typing import Optional

import torch
import torch.optim as optim
from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer

from hive_zero_core.agents.base_expert import BaseExpert


class Agent_Sentinel(BaseExpert):
    """
    Expert 6: The Discriminator (BERT Classifier)
    Classifies payloads as Blocked (0) or Allowed (1).
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        model_name: str = "prajjwal1/bert-tiny",
        hidden_dim: int = 64,
    ):
        super().__init__(observation_dim, action_dim, name="Sentinel", hidden_dim=hidden_dim)
        self.model_name = model_name

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=2
            )
        except Exception as e:
            self.logger.error(f"Failed to load Sentinel model {model_name}: {e}")
            raise e

    def _forward_impl(
        self, x: torch.Tensor, context: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Dynamic Shape Adapter: Ensure input fits BERT constraints
        # BERT expects [Batch, SeqLen] of Long integers (Token IDs) OR [Batch, SeqLen, EmbDim]

        vocab_size = self.model.config.vocab_size
        max_len = 512

        if x.dim() == 2:
            # [Batch, Features] -> Treat as raw inputs or project?
            # If float, we can't use as input_ids directly.
            # Project floats to Token IDs via simple quantization/hashing
            if x.dtype == torch.float:
                # Quantize features to fake token IDs
                input_ids = (torch.abs(x) * 1000).long() % vocab_size
            else:
                input_ids = x.long()

            # Clamp to vocab size
            input_ids = torch.clamp(input_ids, 0, vocab_size - 1)

            # Ensure max len
            if input_ids.size(1) > max_len:
                input_ids = input_ids[:, :max_len]

            outputs = self.model(input_ids=input_ids)

        elif x.dim() == 3:
            # [Batch, Seq, Emb] -> inputs_embeds
            # Adapter: Project dimensions if needed
            # BERT tiny hidden size is 128
            target_emb_dim = 128

            # Simple projection if dims mismatch
            if x.size(-1) != target_emb_dim:
                # We need a linear projection, but since this is inference,
                # we pad or slice for stability.
                # Use base helper
                x_reshaped = x.view(x.size(0) * x.size(1), -1)
                x_adapted = self.ensure_dimension(x_reshaped, target_emb_dim)
                x = x_adapted.view(x.size(0), x.size(1), target_emb_dim)

            outputs = self.model(inputs_embeds=x)

        return outputs.logits


class Agent_PayloadGen(BaseExpert):
    """
    Expert 4: Payload Generator (Seq2Seq)
    Generates raw exploit strings from vulnerability context.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        model_name: str = "t5-small",
        hidden_dim: int = 64,
    ):
        super().__init__(observation_dim, action_dim, name="PayloadGen", hidden_dim=hidden_dim)
        self.model_name = model_name

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        except Exception as e:
            self.logger.error(f"Failed to load PayloadGen model {model_name}: {e}")
            raise e

    def _forward_impl(
        self, x: torch.Tensor, context: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Input x: encoded context token IDs [batch, seq_len]
        max_len = 128
        vocab_size = self.model.config.vocab_size

        # Hardening: Input adaptation
        if x.dtype == torch.float:
            # Quantize
            input_ids = (torch.abs(x) * 100).long() % vocab_size
        else:
            input_ids = x.long()

        input_ids = torch.clamp(input_ids, 0, vocab_size - 1)

        outputs = self.model.generate(input_ids, max_length=max_len, do_sample=True)

        # Output is [Batch, SeqLen].
        # Action dim is fixed. Pad/Truncate output to match action_dim.
        return self.ensure_dimension(outputs.float(), self.action_dim)


class Agent_Mutator(BaseExpert):
    """
    Expert 5: Mutator (PPO / Optimizer)
    Iteratively obfuscates payload to evade Sentinel.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        sentinel_expert: BaseExpert,
        generator_expert: BaseExpert,
        hidden_dim: int = 64,
    ):
        super().__init__(observation_dim, action_dim, name="Mutator", hidden_dim=hidden_dim)

        self.sentinel = sentinel_expert
        self.generator = generator_expert

    def _forward_impl(
        self, x: torch.Tensor, context: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Inference-Time Search Loop.
        """
        # Hardening: Check dependencies active status
        # Ideally, we force them active for this calculation

        # 1. Initial Generation
        with torch.no_grad():
            # Use generator's logic but handle dims manually
            # Ensure input fits T5 requirements
            vocab_size = self.generator.model.config.vocab_size
            if x.dtype == torch.float:
                input_ids = (torch.abs(x) * 100).long() % vocab_size
            else:
                input_ids = x.long()
            input_ids = torch.clamp(input_ids, 0, vocab_size - 1)

            initial_token_ids = self.generator.model.generate(
                input_ids, max_length=64, do_sample=True
            )

        # 2. Convert to Embeddings for Optimization
        embed_layer = self.sentinel.model.get_input_embeddings()
        sentinel_vocab_size = self.sentinel.model.config.vocab_size

        # Reshape or clip if needed. BERT max 512.
        initial_token_ids = initial_token_ids.long()
        if initial_token_ids.shape[-1] > 512:
            initial_token_ids = initial_token_ids[:, :512]

        # CLAMP tokens to vocab size
        initial_token_ids = torch.clamp(initial_token_ids, 0, sentinel_vocab_size - 1)

        # Get initial embeddings
        current_embeddings = embed_layer(initial_token_ids).clone().detach()
        current_embeddings.requires_grad_(True)

        optimizer = optim.SGD([current_embeddings], lr=0.1)

        best_embeddings = current_embeddings.clone().detach()
        best_score = -1.0

        k_steps = 2  # Reduced for stability in prototype

        for i in range(k_steps):
            optimizer.zero_grad()

            # Forward through Sentinel (soft embeddings)
            # Ensure shape compatibility inside Sentinel is handled by its new _forward_impl
            outputs = self.sentinel.model(inputs_embeds=current_embeddings)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

            # Score: P(Allowed) -> index 1
            score = probs[:, 1].mean()

            if score.item() > best_score:
                best_score = score.item()
                best_embeddings = current_embeddings.clone().detach()

            # Maximize P(Allowed)
            loss = -torch.log(probs[:, 1] + 1e-8).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([current_embeddings], max_norm=1.0)
            optimizer.step()

        # Output needs to match action_dim (128).
        # Flatten embeddings or return token IDs?
        # Usually Mutator returns the optimized payload sequence.
        # We assume action_dim expects a sequence or a flattened embedding.
        # Let's return flattened best_embeddings, adapted to size.

        flat = best_embeddings.view(x.size(0), -1)
        return self.ensure_dimension(flat, self.action_dim)
