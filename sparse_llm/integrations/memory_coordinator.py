"""Unified memory coordinator for vLLM and expert cache."""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class UnifiedMemoryCoordinator:
    """Coordinates GPU memory allocation between expert cache and vLLM KV cache.

    Prevents OOM conflicts by establishing static memory budgets at startup.

    Memory Allocation Strategy:
    - Total GPU → 85% usable (15% safety buffer)
    - Usable → 40% expert cache, 50% KV cache, 10% overhead

    This prevents both systems from competing for the same memory.
    """

    # Static allocation ratios
    EXPERT_CACHE_RATIO = 0.40  # 40% of usable GPU for experts
    KV_CACHE_RATIO = 0.50      # 50% of usable GPU for KV cache
    OVERHEAD_RATIO = 0.10      # 10% of usable GPU for activations
    SAFETY_MARGIN = 0.85       # Use 85% of total GPU memory

    def __init__(self, total_gpu_bytes: int):
        """Initialize memory coordinator.

        Args:
            total_gpu_bytes: Total GPU memory in bytes from torch.cuda
        """
        self.total_gpu = total_gpu_bytes
        self.usable_gpu = int(total_gpu_bytes * self.SAFETY_MARGIN)

        # Pre-allocate static budgets
        self.expert_cache_budget = int(self.usable_gpu * self.EXPERT_CACHE_RATIO)
        self.kv_cache_budget = int(self.usable_gpu * self.KV_CACHE_RATIO)
        self.overhead_budget = int(self.usable_gpu * self.OVERHEAD_RATIO)

        logger.info("="*60)
        logger.info("Unified Memory Coordination Initialized")
        logger.info("="*60)
        logger.info(f"Total GPU Memory: {self.total_gpu / 1e9:.2f} GB")
        logger.info(f"Usable GPU Memory: {self.usable_gpu / 1e9:.2f} GB (85%)")
        logger.info(f"Expert Cache Budget: {self.expert_cache_budget / 1e9:.2f} GB (40%)")
        logger.info(f"KV Cache Budget: {self.kv_cache_budget / 1e9:.2f} GB (50%)")
        logger.info(f"Overhead Budget: {self.overhead_budget / 1e9:.2f} GB (10%)")
        logger.info("="*60)

    def get_vllm_config(self) -> Dict[str, Any]:
        """Return vLLM engine configuration respecting memory budget.

        Returns:
            Configuration dict for AsyncEngineArgs with:
            - gpu_memory_utilization: Fraction of total GPU for vLLM
            - max_num_seqs: Maximum batch size
            - enforce_eager: Whether to disable CUDA graphs
        """
        # Set very low GPU utilization (1%) to prevent vLLM from loading weights
        # We inject pre-loaded weights after engine creation
        gpu_memory_utilization = 0.01

        return {
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_num_seqs": self._estimate_max_batch_size(),
            "enforce_eager": True,  # Phase 1: disable CUDA graphs
            "trust_remote_code": True,
        }

    def get_expert_cache_config(self) -> Dict[str, int]:
        """Return expert cache configuration.

        Returns:
            Configuration dict with gpu_cache_bytes and cpu_cache_bytes
        """
        # Estimate CPU budget (typically 10x GPU budget available)
        cpu_budget = self.expert_cache_budget * 10

        return {
            "gpu_cache_bytes": self.expert_cache_budget,
            "cpu_cache_bytes": cpu_budget,
        }

    def _estimate_max_batch_size(self) -> int:
        """Estimate safe maximum batch size for available KV cache.

        Rule of thumb: Each request needs ~128MB KV cache at 2048 context length.

        Returns:
            Maximum number of concurrent sequences (between 8 and 32)
        """
        bytes_per_request = 128 * 1024 * 1024  # 128MB per request
        estimated_batch_size = self.kv_cache_budget // bytes_per_request

        # Clamp to reasonable range
        return max(8, min(32, estimated_batch_size))
