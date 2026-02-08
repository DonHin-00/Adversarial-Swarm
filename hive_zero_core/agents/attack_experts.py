import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, Tuple, List
from transformers import AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoTokenizer
from hive_zero_core.agents.base_expert import BaseExpert
import random

class Agent_Sentinel(BaseExpert):
    """
    Expert 6: The Discriminator (BERT Classifier + History)
    Classifies payloads as Blocked (0) or Allowed (1).
    Maintains a history of recent alerts to simulate IDS threshold behavior.
    """
    def __init__(self, observation_dim: int, action_dim: int, model_name: str = "prajjwal1/bert-tiny", hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="Sentinel", hidden_dim=hidden_dim)
        self.model_name = model_name
        self.hidden_dim = hidden_dim # Explicitly save

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

            # Additional Alert History Component (Simulated IDS state)
            # A simple GRU to track 'suspicion level' over time
            # Input: CLS embedding (from model hidden size)
            bert_hidden_size = self.model.config.hidden_size
            self.history_gru = nn.GRU(bert_hidden_size, hidden_dim, batch_first=True)
            self.alert_threshold_head = nn.Linear(hidden_dim, 1) # Probability modifier

        except Exception as e:
            self.logger.error(f"Failed to load Sentinel model {model_name}: {e}")
            raise e

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: Input payload. [batch, seq_len] (tokens) or [batch, seq_len, dim] (embeddings)

        # 1. Base Classification (Stateless WAF)
        if x.dim() == 3 and x.shape[-1] > 1:
             outputs = self.model(inputs_embeds=x, output_hidden_states=True)
        else:
             outputs = self.model(input_ids=x.long(), output_hidden_states=True)

        logits = outputs.logits # [batch, 2]

        # 2. History/Context Modulation (Stateful IDS)
        # Extract CLS token embedding from last hidden state
        # BERT output hidden_states is tuple. Last one is usually [-1].
        # Shape: [batch, seq_len, hidden_size]
        cls_embedding = outputs.hidden_states[-1][:, 0, :] # [batch, bert_hidden]

        # Assume context is previous hidden state? Or just random noise for prototype?
        # Let's check context. If None, init new.
        if context is not None and context.shape[-1] == self.hidden_dim:
             h_in = context.unsqueeze(0) # [1, batch, gru_hidden]
        else:
             h_in = torch.zeros(1, x.size(0), self.hidden_dim, device=x.device)

        # Update history with current observation (CLS)
        # GRU expects [batch, seq_len, input_size]. Treat current CLS as one step.
        _, h_out = self.history_gru(cls_embedding.unsqueeze(1), h_in)

        # Calculate 'Suspicion Modifier'
        suspicion = torch.sigmoid(self.alert_threshold_head(h_out.squeeze(0))) # [batch, 1]

        # If suspicion is high, probability of 'Allowed' (index 1) decreases.
        base_probs = torch.softmax(logits, dim=-1)

        # Modulate probabilities
        # Return combined probs: [P_blocked, P_allowed]
        # Ensure sum to 1?
        p_allowed_raw = base_probs[:, 1:2]
        p_allowed_mod = p_allowed_raw * (1.0 - 0.5 * suspicion) # Dampen allow prob by up to 50%
        p_blocked_mod = 1.0 - p_allowed_mod

        return torch.cat([p_blocked_mod, p_allowed_mod], dim=1)

class Agent_PayloadGen(BaseExpert):
    """
    Expert 4: Payload Generator (Context-Aware Seq2Seq)
    Takes structured vulnerability context and generates specific exploit payloads.
    """
    def __init__(self, observation_dim: int, action_dim: int, model_name: str = "t5-small", hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="PayloadGen", hidden_dim=hidden_dim)
        self.model_name = model_name

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

            # Context Encoder (Project observation vector to T5 embedding space)
            # Obs dim -> T5 d_model (512 for small)
            t5_dim = self.model.config.d_model
            self.context_projection = nn.Linear(observation_dim, t5_dim)

        except Exception as e:
            self.logger.error(f"Failed to load PayloadGen model {model_name}: {e}")
            raise e

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: Observation vector [batch, obs_dim] representing CVE/Target info

        # 1. Prepare Encoder Inputs (Soft Prompts)
        # Project observation to embedding space
        # T5 encoder expects inputs_embeds [batch, seq_len, dim]
        soft_prompts = self.context_projection(x).unsqueeze(1) # [batch, 1, d_model]

        # We need to provide dummy input_ids or attention_mask usually if passing inputs_embeds via generate?
        # HF generate() with encoder_outputs is cleaner.

        # Run encoder manually
        encoder_outputs_obj = self.model.get_encoder()(inputs_embeds=soft_prompts)

        # Generate using the encoded states
        # encoder_outputs argument in generate expects ModelOutput or tuple
        outputs = self.model.generate(
            encoder_outputs=encoder_outputs_obj,
            max_length=64,
            do_sample=True,
            temperature=0.8
        )

        return outputs

class Agent_Mutator(BaseExpert):
    """
    Expert 5: Mutator (Hybrid Optimizer)
    Combines Gradient-based embedding optimization with Discrete Mutation heuristics.
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
        # x is observation for Generator
        # Generator now expects [batch, obs_dim], not tokens. Ensure x is correct shape.
        with torch.no_grad():
            initial_token_ids = self.generator._forward_impl(x, context)

        # Ensure vocab limits
        vocab_size = self.sentinel.model.config.vocab_size
        initial_token_ids = torch.clamp(initial_token_ids.long(), 0, vocab_size - 1)

        # 2. Gradient-Based Optimization Phase
        embed_layer = self.sentinel.model.get_input_embeddings()

        # Handling dimensionality mismatch between T5 and BERT
        # T5 outputs might be shorter/longer than BERT max pos
        if initial_token_ids.shape[-1] > 512:
             initial_token_ids = initial_token_ids[:, :512]

        current_embeddings = embed_layer(initial_token_ids).clone().detach()
        current_embeddings.requires_grad_(True)

        # Optimizer for this specific instance
        optimizer = optim.SGD([current_embeddings], lr=0.1)

        best_embeddings = current_embeddings.clone().detach()
        best_score = -1.0

        k_steps = 5

        for i in range(k_steps):
            optimizer.zero_grad()

            # Forward Sentinel
            # Sentinel returns PROBS directly now
            probs = self.sentinel._forward_impl(current_embeddings, context=None)

            # Score: P(Allowed) -> index 1
            score = probs[:, 1].mean()

            if score.item() > best_score:
                best_score = score.item()
                best_embeddings = current_embeddings.clone().detach()

            if score.item() > 0.95:
                break

            # Loss: Minimize -log(P(Allowed))
            loss = -torch.log(probs[:, 1] + 1e-8).mean()
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_([current_embeddings], max_norm=1.0)

            optimizer.step()

            # 3. Discrete Mutation Phase (Heuristic Noise)
            # Add noise to jump local optima
            if i % 2 == 0:
                with torch.no_grad():
                    noise = torch.randn_like(current_embeddings) * 0.05
                    current_embeddings.add_(noise)

        return best_embeddings
