"""
Router Calculator for Mixture-of-Experts Models

Computes routing decisions to determine which experts process which tokens.
Optimized for GPU execution with support for both batch (prefill) and
single-token (decode) inference modes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class RouterCalculator:
    """
    Compute router decisions for Mixture-of-Experts (MoE) models.

    Determines which experts should process which tokens by computing router
    logits and selecting top-k experts with highest routing probabilities.
    Optimized for GPU execution with support for both batch (prefill) and
    single-token (decode) modes.

    Performance targets:
        - Prefill mode: <500ms for 1k tokens
        - Decode mode: <5ms for 1 token

    Attributes:
        num_experts: Total number of experts in the MoE layer
        top_k: Number of experts to select per token
        router: Linear projection layer for computing router logits
        device: torch.device where computations are performed
    """

    def __init__(
        self,
        router_weights: torch.Tensor | None,
        num_experts: int,
        top_k: int = 2,
        hidden_dim: int | None = None
    ):
        """
        Initialize the RouterCalculator.

        Args:
            router_weights: Weight matrix for router linear projection.
                Shape should be [hidden_dim, num_experts] or a nn.Linear module.
                If None, will be initialized later via set_router_weights().
            num_experts: Total number of experts in the MoE layer.
            top_k: Number of experts to select per token (default: 2).
                Must be <= num_experts.
            hidden_dim: Hidden dimension (required if router_weights is None).

        Raises:
            ValueError: If top_k > num_experts or if router_weights is None without hidden_dim
        """
        if top_k > num_experts:
            raise ValueError(
                f"top_k ({top_k}) cannot be greater than num_experts ({num_experts})"
            )

        self.num_experts = num_experts
        self.top_k = top_k
        self.router_weights = None  # Will be set later

        # Handle both raw tensors and nn.Linear modules
        if router_weights is None:
            # Router weights will be loaded later from checkpoint
            if hidden_dim is None:
                raise ValueError("hidden_dim must be provided when router_weights is None")
            # Create uninitialized router (weights will be loaded from checkpoint)
            self.router = nn.Linear(hidden_dim, num_experts, bias=False)
            self.router_weights = None  # Mark as uninitialized
        elif isinstance(router_weights, nn.Linear):
            self.router = router_weights
            self.router_weights = router_weights.weight.T  # Store for reference
        else:
            # Create linear layer from weights
            # Expect router_weights shape: [hidden_dim, num_experts]
            hidden_dim, out_dim = router_weights.shape
            if out_dim != num_experts:
                raise ValueError(
                    f"router_weights output dimension ({out_dim}) must match "
                    f"num_experts ({num_experts})"
                )

            self.router = nn.Linear(hidden_dim, num_experts, bias=False)
            # PyTorch Linear expects weight shape [out_features, in_features]
            self.router.weight.data = router_weights.T
            self.router_weights = router_weights

        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.router = self.router.to(self.device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        batch_mode: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute expert assignments for input hidden states.

        This method computes router logits, applies softmax normalization,
        and selects the top-k experts with highest routing probabilities
        for each token.

        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_dim].
                For decode mode, seq_len is typically 1.
                For prefill mode, seq_len can be 1k-100k tokens.
            batch_mode: Whether to process in batch mode (prefill) or single-token
                mode (decode). Currently both use the same optimized GPU path.
                This parameter is reserved for future optimizations.

        Returns:
            Tuple of (expert_indices, expert_weights):

            expert_indices: torch.Tensor of shape [batch_size, seq_len, top_k]
                Long tensor containing selected expert IDs (0 to num_experts-1)
                for each token. Experts are sorted by routing weight (descending).

            expert_weights: torch.Tensor of shape [batch_size, seq_len, top_k]
                Float tensor containing normalized routing weights for selected
                experts. Weights sum to 1.0 across the top_k dimension for each token.

        Performance:
            - Batch mode (1k tokens): <500ms target on modern GPUs
            - Single-token mode: <5ms target on modern GPUs

        Example:
            >>> router = RouterCalculator(weights, num_experts=8, top_k=2)
            >>> hidden = torch.randn(1, 100, 768).cuda()  # batch=1, seq=100
            >>> indices, weights = router.forward(hidden)
            >>> indices.shape
            torch.Size([1, 100, 2])
            >>> weights.sum(dim=-1)  # Should be all 1.0
            tensor([[1., 1., 1., ..., 1., 1., 1.]])

        Raises:
            ValueError: If hidden_states doesn't have 3 dimensions
        """
        # Ensure input is on correct device
        hidden_states = hidden_states.to(self.device)

        # Validate input shape
        if hidden_states.dim() != 3:
            raise ValueError(
                f"Expected hidden_states to have 3 dimensions [batch, seq, hidden], "
                f"but got shape {hidden_states.shape}"
            )

        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Compute router logits: [batch, seq, num_experts]
        # This is the most compute-intensive operation
        router_logits = self.router(hidden_states)

        # Apply softmax to get routing probabilities
        # Using dim=-1 to normalize across experts for each token
        routing_probs = F.softmax(router_logits, dim=-1)

        # Select top-k experts per token
        # topk returns (values, indices) where values are sorted in descending order
        expert_weights, expert_indices = torch.topk(
            routing_probs,
            k=self.top_k,
            dim=-1,
            sorted=True
        )

        # Normalize weights to sum to 1.0 across top-k
        # This ensures numerical stability and proper weighted combination
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)

        return expert_indices, expert_weights

    def get_expert_assignments(
        self,
        router_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert router logits directly to expert assignments.

        Useful when logits are already computed elsewhere (e.g., cached from
        a previous computation) and you just need the top-k selection logic.
        This avoids recomputing the router projection.

        Args:
            router_logits: Pre-computed logits of shape [batch, seq, num_experts]

        Returns:
            Tuple of (expert_indices, expert_weights) with same semantics as forward().

            expert_indices: shape [batch, seq, top_k]
            expert_weights: shape [batch, seq, top_k], normalized to sum to 1.0

        Example:
            >>> logits = torch.randn(1, 10, 8)  # batch=1, seq=10, 8 experts
            >>> indices, weights = router.get_expert_assignments(logits)
            >>> indices.shape
            torch.Size([1, 10, 2])
        """
        router_logits = router_logits.to(self.device)

        # Apply softmax
        routing_probs = F.softmax(router_logits, dim=-1)

        # Select top-k
        expert_weights, expert_indices = torch.topk(
            routing_probs,
            k=self.top_k,
            dim=-1,
            sorted=True
        )

        # Normalize weights
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)

        return expert_indices, expert_weights

    def to(self, device: torch.device) -> 'RouterCalculator':
        """
        Move router to specified device.

        Args:
            device: Target device (e.g., torch.device('cuda') or torch.device('cpu'))

        Returns:
            Self for method chaining

        Example:
            >>> router = RouterCalculator(weights, num_experts=8)
            >>> router.to(torch.device('cuda:1'))  # Move to second GPU
        """
        self.device = device
        self.router = self.router.to(device)
        return self

    def set_router_weights(self, weights: torch.Tensor) -> None:
        """
        Set router weights after initialization (for lazy loading from checkpoint).

        Args:
            weights: Weight tensor of shape [hidden_dim, num_experts]
        """
        hidden_dim, out_dim = weights.shape
        if out_dim != self.num_experts:
            raise ValueError(
                f"router_weights output dimension ({out_dim}) must match "
                f"num_experts ({self.num_experts})"
            )

        # Update the router's weights (PyTorch Linear expects transposed format)
        self.router.weight.data = weights.T.to(self.device)
        self.router_weights = weights

    @property
    def weight(self) -> torch.Tensor:
        """
        Access router weight matrix.

        Returns:
            Weight tensor of shape [hidden_dim, num_experts]
        """
        # Return in [hidden_dim, num_experts] format (transpose of internal storage)
        return self.router.weight.T

    def __repr__(self) -> str:
        """String representation of RouterCalculator."""
        return (
            f"RouterCalculator(num_experts={self.num_experts}, "
            f"top_k={self.top_k}, device={self.device})"
        )
