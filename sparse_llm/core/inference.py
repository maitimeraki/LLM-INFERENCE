import logging
from typing import List, Optional, Tuple
import torch
import time
from sparse_llm.cache.expert_cache import ExpertCache
from sparse_llm.storage import StorageBackend, LocalSSDStorage
from sparse_llm.prefetch.pipeline import PrefetchPipeline
from sparse_llm.core.router import RouterPredictor
from sparse_llm.quantization import QuantizationManager

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Main inference engine orchestrating cache, storage, prefetch, and prediction."""

    def __init__(self,
                 model_name: str,
                 device: str = "cuda",
                 max_cached_experts: int = 22,
                 storage_backend: Optional[StorageBackend] = None):
        self.model_name = model_name
        self.device = device
        self.max_cached_experts = max_cached_experts

        # Initialize components
        self.cache = ExpertCache(max_experts=max_cached_experts)
        self.storage = storage_backend or LocalSSDStorage()
        self.prefetch = PrefetchPipeline(num_streams=3)
        self.router_predictor = RouterPredictor(num_experts=256)
        self.quantizer = QuantizationManager(bits=4)

        # Statistics
        self.inference_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.prefetch_hits = 0

    def get_expert(self, expert_id: int) -> torch.Tensor:
        """Load expert from cache or storage, with prefetch integration."""
        # Check cache first
        cached = self.cache.get(expert_id)
        if cached is not None:
            self.cache_hits += 1
            return cached

        self.cache_misses += 1

        # Load from storage
        try:
            weights = self.storage.load_expert(expert_id)
            if weights.device != self.device:
                weights = weights.to(self.device)

            # Add to cache
            self.cache.put(expert_id, weights)
            return weights
        except Exception as e:
            logger.error(f"Failed to load expert {expert_id}: {e}")
            raise

    def prefetch_experts(self, predicted_experts: List[int],
                        confidence_scores: List[float]) -> None:
        """Schedule prefetch for predicted experts."""
        cached = {exp_id for exp_id in predicted_experts if self.cache.contains(exp_id)}
        distribution = self.prefetch.schedule_prefetch(predicted_experts, confidence_scores, cached)

        # Load L0 experts immediately
        for expert_id in predicted_experts[:3]:
            if expert_id not in cached:
                try:
                    weights = self.storage.load_expert(expert_id)
                    if weights.device != self.device:
                        weights = weights.to(self.device)
                    self.cache.put(expert_id, weights)
                    self.prefetch_hits += 1
                except Exception as e:
                    logger.warning(f"Prefetch failed for expert {expert_id}: {e}")

    def forward(self, hidden_state: torch.Tensor,
                router_output: Tuple[List[int], torch.Tensor]) -> torch.Tensor:
        """
        Forward pass: load experts, compute, update predictor.

        Args:
            hidden_state: Input tensor
            router_output: (expert_ids, router_logits)

        Returns:
            Expert output
        """
        self.inference_count += 1
        expert_ids, router_logits = router_output

        # Load experts
        expert_weights = []
        for exp_id in expert_ids:
            weights = self.get_expert(exp_id)
            expert_weights.append(weights)

        # Update predictor with actual routing
        self.router_predictor.update_from_inference(expert_ids, router_logits)

        # Predict next experts and prefetch
        predicted_experts, confidence_scores = self.router_predictor.predict_next_experts(
            expert_ids, top_k=15
        )
        self.prefetch_experts(predicted_experts, confidence_scores)

        # Simple expert combination (average)
        if expert_weights:
            expert_output = torch.stack(expert_weights, dim=0).mean(dim=0)
            return expert_output
        return hidden_state

    def stats(self) -> dict:
        """Return inference statistics."""
        total_accesses = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_accesses * 100) if total_accesses > 0 else 0

        return {
            "inference_count": self.inference_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": hit_rate,
            "prefetch_hits": self.prefetch_hits,
            "cache_stats": self.cache.stats(),
            "prefetch_stats": self.prefetch.stats(),
            "router_stats": self.router_predictor.stats(),
        }

    def clear_cache(self) -> None:
        """Clear expert cache."""
        self.cache.clear()
        logger.info("Cleared inference cache")
