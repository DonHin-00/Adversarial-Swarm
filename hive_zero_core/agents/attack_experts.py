import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional, Dict
from transformers import AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoTokenizer
from hive_zero_core.agents.base_expert import BaseExpert

class SentinelAgent(BaseExpert):
    """
    Expert 6: Stateful Ensemble Discriminator
    Uses a history GRU to model alert thresholds across sequences.
    """
    def __init__(self, observation_dim: int, action_dim: int, model_name: str = "prajjwal1/bert-tiny", hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="Sentinel", hidden_dim=hidden_dim)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.backbone = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=hidden_dim, output_hidden_states=True)

            # Ensemble Heads
            self.head1 = nn.Linear(hidden_dim, 2)
            self.head2 = nn.Linear(hidden_dim, 2)
            self.head3 = nn.Linear(hidden_dim, 2)

            # Stateful Threshold Modeling
            self.history_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        except (ImportError, RuntimeError, OSError) as e:
            self.logger.error(f"Failed to load Sentinel model/tokenizer: {str(e)}")
            raise RuntimeError(f"Sentinel initialization failed: {str(e)}")

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [Batch, Seq, Dim] or [Batch, Seq]
        if x.dim() == 3 and x.shape[-1] > 1:
             outputs = self.backbone(inputs_embeds=x)
        else:
             outputs = self.backbone(input_ids=x.long())

        features = F.relu(outputs.logits) # [Batch, hidden_dim]

        # Stateful refinement
        h_seq = features.unsqueeze(1)
        h_out, _ = self.history_gru(h_seq)
        refined_features = h_out.squeeze(1)

        # Ensemble Voting
        logits1 = self.head1(refined_features)
        logits2 = self.head2(refined_features)
        logits3 = self.head3(refined_features)

        avg_probs = (torch.softmax(logits1, -1) + torch.softmax(logits2, -1) + torch.softmax(logits3, -1)) / 3.0
        return avg_probs

class PayloadGenAgent(BaseExpert):
    """
    Expert 4: RAG-Enhanced Payload Generator
    Retrieves "Exploit Templates" from a VectorDB before generation.
    """
    def __init__(self, observation_dim: int, action_dim: int, model_name: str = "t5-small",
                 hidden_dim: int = 64, rag_db: Optional[Dict[str, torch.Tensor]] = None):
        super().__init__(observation_dim, action_dim, name="PayloadGen", hidden_dim=hidden_dim)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

            # Vector DB (Keys: Embeddings, Values: Template Tokens)
            if rag_db:
                self.db_keys = nn.Parameter(rag_db["keys"], requires_grad=False)
                self.db_values = nn.Parameter(rag_db["values"], requires_grad=False)
            else:
                # Default mock DB
                self.db_keys = nn.Parameter(torch.randn(10, 64), requires_grad=False)
                self.db_values = nn.Parameter(torch.randint(0, 1000, (10, 20)), requires_grad=False)

            self.query_proj = nn.Linear(observation_dim, 64)

        except (ImportError, RuntimeError, OSError) as e:
            self.logger.error(f"Failed to load PayloadGen model/tokenizer: {str(e)}")
            raise RuntimeError(f"PayloadGen initialization failed: {str(e)}")

    def _retrieve(self, query: torch.Tensor) -> torch.Tensor:
        # query: [B, 64]
        scores = torch.matmul(query, self.db_keys.t())
        indices = torch.argmax(scores, dim=1) # [B]

        # Retrieve templates
        templates = self.db_values[indices] # [B, 20]
        return templates

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. RAG Retrieval
        query = self.query_proj(x)
        retrieved_tokens = self._retrieve(query)

        # 2. Generation with Context
        # Using mock encoder outputs to condition generation
        # We simulate this by passing retrieval embedding as 'encoder_outputs'

        # Embed retrieval
        # T5 encoder expects input_ids.
        # We need to run the full encoder stack to get valid encoder_outputs
        encoder_out = self.model.encoder(input_ids=retrieved_tokens.long())

        # Generate
        outputs = self.model.generate(
            encoder_outputs=encoder_out,
            max_length=64
        )

        return outputs

class MutatorAgent(BaseExpert):
    """
    Expert 5: Hybrid Search Mutator
    Implements Inference-Time Search using gradient descent + noise injection.
    Optimizes payloads to evade SentinelAgent.
    """
    def __init__(self, observation_dim: int, action_dim: int, sentinel_expert: BaseExpert, generator_expert: BaseExpert, hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="Mutator", hidden_dim=hidden_dim)
        self.sentinel = sentinel_expert
        self.generator = generator_expert
    
    def _validate_dependencies(self) -> None:
        """
        Validates that required dependencies (tokenizers, models) are initialized.
        
        Raises:
            RuntimeError: If any required dependency is not properly initialized
        """
        if not hasattr(self.generator, 'tokenizer') or self.generator.tokenizer is None:
            raise RuntimeError(
                "Generator tokenizer not initialized. Ensure PayloadGenAgent "
                "was constructed successfully before using MutatorAgent."
            )
        
        if not hasattr(self.sentinel, 'tokenizer') or self.sentinel.tokenizer is None:
            raise RuntimeError(
                "Sentinel tokenizer not initialized. Ensure SentinelAgent "
                "was constructed successfully before using MutatorAgent."
            )
        
        if not hasattr(self.sentinel, 'backbone') or self.sentinel.backbone is None:
            raise RuntimeError(
                "Sentinel backbone not initialized. Ensure SentinelAgent "
                "was constructed successfully before using MutatorAgent."
            )

    def _inner_loop_search(self, embedding: torch.Tensor, iterations: int = 5) -> torch.Tensor:
        # embedding: [1, seq, dim]
        optimized_emb = embedding.clone().detach().requires_grad_(True)
        optimizer = optim.SGD([optimized_emb], lr=0.01)

        for _ in range(iterations):
            optimizer.zero_grad()
            # We want to MAXIMIZE probability of evasion (P(Allowed))
            # Call Sentinel's ungated implementation to ensure gradients flow through it
            probs = self.sentinel._forward_impl(optimized_emb)
            loss = -torch.log(probs[:, 1] + 1e-8).mean()
            loss.backward()
            optimizer.step()

            # Discrete Noise Injection (Hybrid)
            with torch.no_grad():
                noise = torch.randn_like(optimized_emb) * 0.05
                optimized_emb.add_(noise)

        return optimized_emb.detach()

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Validate dependencies before proceeding (raises RuntimeError if invalid)
        self._validate_dependencies()
        
        # 1. Base Generation
        with torch.no_grad():
            # Bypass gating for internal dependencies to avoid zero tensors
            gen_out = self.generator._forward_impl(x, context)
            
            # Decode T5 tokens to text and re-encode to BERT tokens to avoid vocab mismatch
            # gen_out shape: [Batch, Seq]
            gen_text = self.generator.tokenizer.batch_decode(gen_out, skip_special_tokens=True)
            
            # Re-encode for Sentinel (BERT)
            sentinel_inputs = self.sentinel.tokenizer(gen_text, return_tensors="pt", padding=True, truncation=True).to(x.device)
            initial_token_ids = sentinel_inputs["input_ids"]

        embed_layer = self.sentinel.backbone.get_input_embeddings()
        current_embeddings = embed_layer(initial_token_ids).detach()

        # 2. Inference-Time Search
        if current_embeddings.size(0) == 1:
            return self._inner_loop_search(current_embeddings)
        return current_embeddings
