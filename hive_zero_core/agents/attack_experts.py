import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, Tuple
from transformers import AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoTokenizer
from hive_zero_core.agents.base_expert import BaseExpert

class Agent_Sentinel(BaseExpert):
    """
    Expert 6: The Discriminator (BERT Classifier)
    Classifies payloads as Blocked (0) or Allowed (1).
    """
    def __init__(self, observation_dim: int, action_dim: int, model_name: str = "prajjwal1/bert-tiny", hidden_dim: int = 64):
        # Using bert-tiny to reduce memory footprint for testing
        super().__init__(observation_dim, action_dim, name="Sentinel", hidden_dim=hidden_dim)
        self.model_name = model_name

        # Hardening: Load model with robust error handling
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        except Exception as e:
            self.logger.error(f"Failed to load Sentinel model {model_name}: {e}")
            raise e

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() == 3 and x.shape[-1] > 1:
             outputs = self.model(inputs_embeds=x)
        else:
             outputs = self.model(input_ids=x.long())

        return outputs.logits

class Agent_PayloadGen(BaseExpert):
    """
    Expert 4: Payload Generator (Seq2Seq)
    Generates raw exploit strings from vulnerability context.
    """
    def __init__(self, observation_dim: int, action_dim: int, model_name: str = "t5-small", hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="PayloadGen", hidden_dim=hidden_dim)
        self.model_name = model_name

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        except Exception as e:
            self.logger.error(f"Failed to load PayloadGen model {model_name}: {e}")
            raise e

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Input x: encoded context token IDs [batch, seq_len]

        max_len = 128
        if x.dtype == torch.float:
            input_ids = x.long()
        else:
            input_ids = x

        outputs = self.model.generate(input_ids, max_length=max_len, do_sample=True)
        return outputs

class Agent_Mutator(BaseExpert):
    """
    Expert 5: Mutator (PPO / Optimizer)
    Iteratively obfuscates payload to evade Sentinel.
    """
    def __init__(self, observation_dim: int, action_dim: int, sentinel_expert: BaseExpert, generator_expert: BaseExpert, hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="Mutator", hidden_dim=hidden_dim)

        self.sentinel = sentinel_expert
        self.generator = generator_expert

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Inference-Time Search Loop.
        """
        # 1. Initial Generation
        # Assume x is context for Generator [batch, seq_len]
        with torch.no_grad():
            # Generate initial tokens
            initial_token_ids = self.generator.model.generate(x.long(), max_length=64, do_sample=True) # Clip length

        # 2. Convert to Embeddings for Optimization
        embed_layer = self.sentinel.model.get_input_embeddings()
        vocab_size = self.sentinel.model.config.vocab_size

        # Reshape or clip if needed. BERT max 512.
        initial_token_ids = initial_token_ids.long()
        if initial_token_ids.shape[-1] > 512:
            initial_token_ids = initial_token_ids[:, :512]

        # CLAMP tokens to vocab size to prevent index out of range
        initial_token_ids = torch.clamp(initial_token_ids, 0, vocab_size - 1)

        # Get initial embeddings
        current_embeddings = embed_layer(initial_token_ids).clone().detach()
        current_embeddings.requires_grad_(True)

        # Optimizer for this specific instance
        optimizer = optim.SGD([current_embeddings], lr=0.1)

        best_embeddings = current_embeddings.clone().detach()
        best_score = -1.0

        k_steps = 5
        patience = 2
        no_improve_count = 0

        for i in range(k_steps):
            optimizer.zero_grad()

            # Forward through Sentinel (soft embeddings)
            outputs = self.sentinel.model(inputs_embeds=current_embeddings)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

            # Score: P(Allowed) -> index 1
            score = probs[:, 1].mean()

            if score.item() > best_score:
                best_score = score.item()
                best_embeddings = current_embeddings.clone().detach()
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= patience:
                break

            if score.item() > 0.99:
                break

            # Maximize P(Allowed)
            loss = -torch.log(probs[:, 1] + 1e-8).mean()
            loss.backward()

            torch.nn.utils.clip_grad_norm_([current_embeddings], max_norm=1.0)

            optimizer.step()

        return best_embeddings
