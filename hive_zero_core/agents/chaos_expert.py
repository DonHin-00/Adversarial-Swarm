import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config
from hive_zero_core.agents.base_expert import BaseExpert
from typing import Optional, Dict

class Agent_Chaos(BaseExpert):
    """
    Expert 13: Protocol Fuzzer (GPT-2)
    Generates structurally valid but semantically mutated packets (Fuzzing).
    Learns protocol grammar from byte sequences.
    """
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="Chaos", hidden_dim=hidden_dim)

        # Tiny GPT-2 configuration for byte-level modeling
        config = GPT2Config(
            vocab_size=256, # Byte vocab
            n_positions=1024,
            n_embd=64, # Small for efficiency
            n_layer=2,
            n_head=2
        )
        self.model = GPT2LMHeadModel(config)

        # Action Head: Project hidden state to fuzzing parameters?
        # Or just use the generated sequence as the packet.
        # Action dim usually unused for generation experts (return sequence).

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> Any:
        # x: Seed Packet (Byte sequence) [Batch, Seq]

        # Generate fuzzed sequence
        # Temperature > 1.0 for Chaos

        outputs = self.model.generate(
            input_ids=x.long(),
            max_new_tokens=64,
            do_sample=True,
            temperature=1.5, # High chaos
            pad_token_id=0
        )

        return outputs
