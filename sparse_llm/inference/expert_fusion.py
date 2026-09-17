"""Expert fusion cache for pre-computed fused expert weights."""

from collections import Counter
from typing import Optional

import torch


class ExpertFusionCache:
    """Cache fused expert weights for common routing patterns.

    For path {E3, E7} with weights [0.7, 0.3]:
        W_fused = 0.7 * W3 + 0.3 * W7
        result = x @ W_fused  # Single matmul instead of two
    """

    def __init__(self, max_fused_paths: int = 1000):
        self.max_fused_paths = max_fused_paths
        self._fused: dict[int, dict[str, torch.Tensor]] = {}
        self._access_count: Counter[int] = Counter()

    def get_fused(
        self,
        path_fp: int,
        expert_ids: list[int],
        weights: list[float],
        tiled_weights: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Get or compute fused weights for expert path.

        Args:
            path_fp: Fingerprint of the routing path
            expert_ids: List of expert IDs in this path
            weights: Routing weights for each expert
            tiled_weights: Dict of tiled weight tensors

        Returns:
            Dict of fused weight tensors {w1, w2, w3}
        """
        self._access_count[path_fp] += 1

        if path_fp not in self._fused:
            fused = {}
            for key in ["w1", "w2", "w3"]:
                if key in tiled_weights and tiled_weights[key].shape[0] > 1:
                    base = tiled_weights[key][0]
                    fused_tensor = torch.zeros_like(base)
                    for expert_id, weight in zip(expert_ids, weights):
                        if expert_id < tiled_weights[key].shape[0]:
                            fused_tensor = fused_tensor + weight * tiled_weights[key][expert_id]
                    fused[key] = fused_tensor
            self._fused[path_fp] = fused

            if len(self._fused) > self.max_fused_paths:
                lru_fp = min(self._access_count, key=self._access_count.get)
                self._fused.pop(lru_fp, None)
                self._access_count.pop(lru_fp, None)

        return self._fused[path_fp]

    def invalidate(self, path_fp: Optional[int] = None) -> None:
        """Invalidate cache entry or all entries."""
        if path_fp is not None:
            self._fused.pop(path_fp, None)
            self._access_count.pop(path_fp, None)
        else:
            self._fused.clear()
            self._access_count.clear()

    def get_stats(self) -> dict:
        """Return cache statistics."""
        return {
            "num_entries": len(self._fused),
            "max_entries": self.max_fused_paths,
            "total_accesses": sum(self._access_count.values()),
        }


__all__ = ["ExpertFusionCache"]
