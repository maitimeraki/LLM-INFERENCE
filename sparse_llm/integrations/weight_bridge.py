"""Weight bridge between vLLM and LoadedWeightState."""

import logging
import torch
from typing import Tuple, Dict
from sparse_llm.loading.loaded_weight_state import LoadedWeightState

logger = logging.getLogger(__name__)


class SparseMoEWeightBridge:
    """Bridge that provides vLLM with weights from LoadedWeightState.

    This is the critical integration point that prevents duplicate weight loading.
    Instead of vLLM loading weights from HuggingFace, it retrieves them from
    the pre-loaded LoadedWeightState created by FourPhaseOrchestrator.

    For shared weights: Direct tensor lookup (zero-copy)
    For expert weights: Lookup from 3-tier cache with automatic promotion
    """

    def __init__(self, loaded_state: LoadedWeightState):
        """Initialize weight bridge.

        Args:
            loaded_state: Pre-loaded weights from FourPhaseOrchestrator
        """
        self.loaded_state = loaded_state
        self.expert_cache = loaded_state.expert_cache
        self.shared_weights = loaded_state.shared_weights
        self.model_info = loaded_state.model_info

        # Cache for parsed weight names (avoid repeated parsing)
        self._name_cache: Dict[str, Tuple[str, int, int]] = {}

        logger.info(f"SparseMoEWeightBridge initialized for {self.model_info.model_id}")
        logger.info(f"  MoE model: {self.model_info.is_moe}")
        logger.info(f"  Shared weights: {len(self.shared_weights)} tensors")
        if self.model_info.is_moe:
            logger.info(f"  Expert cache: {self.model_info.num_layers}L x {self.model_info.num_experts}E")

    def get_weight(self, name: str) -> torch.Tensor:
        """Get weight tensor by parameter name.

        This method is called by vLLM during model initialization.

        Args:
            name: Full parameter name (e.g., "model.layers.5.block_sparse_moe.experts.3.w1.weight")

        Returns:
            Weight tensor on GPU

        Raises:
            KeyError: If weight not found in LoadedWeightState
        """
        # Route to appropriate handler based on weight type
        if ".experts." in name:
            return self._get_expert_weight(name)
        else:
            return self._get_shared_weight(name)

    def _get_shared_weight(self, name: str) -> torch.Tensor:
        """Get shared (non-expert) weight.

        Args:
            name: Weight name

        Returns:
            Weight tensor on GPU
        """
        if name not in self.shared_weights:
            available = list(self.shared_weights.keys())[:10]  # Show first 10
            raise KeyError(
                f"Shared weight '{name}' not found in LoadedWeightState. "
                f"Available weights (first 10): {available}"
            )

        weight_tensor = self.shared_weights[name]

        # Ensure on GPU
        if weight_tensor.device.type != "cuda":
            weight_tensor = weight_tensor.cuda()

        return weight_tensor

    def _get_expert_weight(self, name: str) -> torch.Tensor:
        """Get expert weight from 3-tier cache.

        Args:
            name: Expert weight name

        Returns:
            Weight tensor on GPU (with automatic cache promotion if needed)
        """
        # Parse name (with caching)
        if name not in self._name_cache:
            layer_id, expert_id, weight_key = self._parse_expert_name(name)
            self._name_cache[name] = (weight_key, layer_id, expert_id)
        else:
            weight_key, layer_id, expert_id = self._name_cache[name]

        # Get expert weights from cache (auto-promotes to GPU if needed)
        expert_weights, tier = self.expert_cache.get(layer_id, expert_id)

        # Extract specific weight from expert dict
        if weight_key not in expert_weights:
            available = list(expert_weights.keys())
            raise KeyError(
                f"Weight '{weight_key}' not found in expert (layer={layer_id}, expert={expert_id}). "
                f"Available weights: {available}"
            )

        weight_tensor = expert_weights[weight_key]

        # Ensure on GPU (cache might return CPU tensor)
        if weight_tensor.device.type != "cuda":
            weight_tensor = weight_tensor.cuda()

        return weight_tensor

    def _parse_expert_name(self, name: str) -> Tuple[int, int, str]:
        """Parse expert weight name into components.

        Handles standard MoE naming: "model.layers.{L}.block_sparse_moe.experts.{E}.{weight}"

        Args:
            name: Full weight name

        Returns:
            Tuple of (layer_id, expert_id, weight_key)

        Examples:
            "model.layers.5.block_sparse_moe.experts.3.w1.weight" → (5, 3, "w1.weight")
            "model.layers.10.experts.7.w2.weight" → (10, 7, "w2.weight")
        """
        parts = name.split(".")

        # Find layer index
        try:
            layer_idx_pos = parts.index("layers") + 1
            layer_id = int(parts[layer_idx_pos])
        except (ValueError, IndexError) as e:
            raise ValueError(f"Cannot parse layer_id from weight name: {name}") from e

        # Find expert index
        try:
            expert_idx_pos = parts.index("experts") + 1
            expert_id = int(parts[expert_idx_pos])
        except (ValueError, IndexError) as e:
            raise ValueError(f"Cannot parse expert_id from weight name: {name}") from e

        # Weight key is everything after "experts.{id}."
        weight_key = ".".join(parts[expert_idx_pos + 1:])

        return layer_id, expert_id, weight_key
