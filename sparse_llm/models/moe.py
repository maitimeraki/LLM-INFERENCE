import logging
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


class RouterGate(nn.Module):
    """Trainable gating network for expert routing."""

    def __init__(self, input_dim: int, num_experts: int, top_k: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k

        self.gate = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_experts),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route token x to top-k experts.

        Args:
            x: token embedding [batch_size, input_dim]

        Returns:
            (gate_logits, expert_indices, gate_probs)
        """
        gate_logits = self.gate(x)
        gate_probs = torch.softmax(gate_logits, dim=-1)

        top_probs, expert_indices = torch.topk(gate_probs, self.top_k, dim=-1)

        return gate_logits, expert_indices, top_probs


class SparseMoELayer(nn.Module):
    """Sparse MoE layer: route tokens to top-k experts."""

    def __init__(self, input_dim: int, num_experts: int, expert_dim: int, top_k: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.top_k = top_k

        self.router = RouterGate(input_dim, num_experts, top_k)
        self.experts = nn.ModuleList([
            nn.Linear(input_dim, expert_dim) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor, expert_ids: Optional[List[int]] = None) -> torch.Tensor:
        """
        Process input through routed experts.

        Args:
            x: input tensor [seq_len, batch_size, input_dim]
            expert_ids: list of expert IDs to use (for caching/prefetch)

        Returns:
            output: [seq_len, batch_size, expert_dim]
        """
        seq_len, batch_size, _ = x.shape
        output = torch.zeros(seq_len, batch_size, self.expert_dim, device=x.device, dtype=x.dtype)

        for t in range(seq_len):
            token = x[t]
            _, selected_experts, gate_probs = self.router(token)

            for batch_idx in range(batch_size):
                expert_out = torch.zeros(self.expert_dim, device=x.device, dtype=x.dtype)
                for k in range(self.top_k):
                    if selected_experts.device.type != "meta":
                        expert_id = int(selected_experts[batch_idx, k].item())
                        expert_prob = gate_probs[batch_idx, k]
                        if expert_id < len(self.experts):
                            expert_out += expert_prob * self.experts[expert_id](token[batch_idx])

                output[t, batch_idx] = expert_out

        return output

    def get_routed_experts(self, x: torch.Tensor) -> List[int]:
        """Return list of experts needed for given input."""
        _, expert_indices, _ = self.router(x)
        experts = expert_indices.flatten().unique().tolist()
        return sorted(experts)
