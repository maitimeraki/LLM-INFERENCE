"""Weight loader with streaming and direct device placement."""

from __future__ import annotations
import re
from pathlib import Path
from typing import Callable, Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

from sparse_llm.loading.placement_plan import PlacementPlan
from sparse_llm.loading.expert_cache import ExpertCache
from sparse_llm.loading.loaded_weight_state import LoadedWeightState
from sparse_llm.models.shared_weight_loader import SharedWeightPlacer


class WeightLoader:
    """Load model weights with streaming and direct device placement."""

    def load(
        self,
        model_id: str,
        plan: PlacementPlan,
        progress_callback: Callable[[str], None] | None = None
    ) -> LoadedWeightState:
        """Load weights according to placement plan.

        Args:
            model_id: Hugging Face model ID or local path
            plan: Placement plan from Phase 2
            progress_callback: Optional callback for progress updates

        Returns:
            LoadedWeightState with weights loaded and cache initialized
        """
        def log(message: str):
            if progress_callback:
                progress_callback(message)

        log("⏳ Loading model weights...")

        # Step 1: Download model if needed
        model_path = self._ensure_model_downloaded(model_id, log)

        # Step 2: Load shared weights to target device
        log("[1/4] Loading shared weights to GPU...")
        shared_weights = self._load_shared_weights(model_path, plan, log)

        # Step 3: Initialize expert cache
        log("[2/4] Initializing expert cache...")
        expert_cache = self._initialize_expert_cache(model_path, plan, log)

        # Step 4: Preload hot experts to GPU
        log("[3/4] Loading hot experts to GPU...")
        self._preload_hot_experts(model_path, plan, expert_cache, log)

        # Step 5: Create loaded state
        log("[4/4] Finalizing weight state...")
        state = LoadedWeightState(
            shared_weights=shared_weights,
            expert_cache=expert_cache,
            model_info=plan.model_info,
            placement_plan=plan
        )

        log("✅ Model weights loaded successfully")

        return state

    def _ensure_model_downloaded(self, model_id: str, log: Callable[[str], None]) -> Path:
        """Ensure model is available locally."""
        # Check if it's already a local path
        local_path = Path(model_id)
        if local_path.exists():
            return local_path

        # Download from Hugging Face
        log(f"Downloading model from Hugging Face: {model_id}")
        downloaded_path = snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json"])
        return Path(downloaded_path)

    def _load_shared_weights(
        self,
        model_path: Path,
        plan: PlacementPlan,
        log: Callable[[str], None]
    ) -> dict[str, torch.Tensor]:
        """Load shared weights to target device."""
        shared_weights = {}
        device = plan.shared_device

        # Find safetensors files
        safetensors_files = list(model_path.glob("*.safetensors"))
        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors files found in {model_path}")

        # Load shared weights from all files
        for st_file in safetensors_files:
            with safe_open(st_file, framework="pt", device=device) as f:
                for key in f.keys():
                    if self._is_shared_weight(key):
                        shared_weights[key] = f.get_tensor(key)

        log(f"   ✓ Loaded {len(shared_weights)} shared weight tensors ({self._format_bytes(self._total_bytes(shared_weights))})")

        return shared_weights

    def _initialize_expert_cache(
        self,
        model_path: Path,
        plan: PlacementPlan,
        log: Callable[[str], None]
    ) -> ExpertCache:
        """Initialize expert cache with storage loader."""
        # Create storage loader function
        def storage_loader(layer_id: int, expert_id: int) -> dict[str, torch.Tensor]:
            return self._load_expert_from_storage(model_path, layer_id, expert_id)

        cache = ExpertCache(
            gpu_slots=plan.hot_expert_slots,
            cpu_slots=plan.warm_expert_slots,
            expert_bytes=plan.model_info.expert_weight_bytes,
            storage_loader=storage_loader
        )

        return cache

    def _preload_hot_experts(
        self,
        model_path: Path,
        plan: PlacementPlan,
        expert_cache: ExpertCache,
        log: Callable[[str], None]
    ) -> None:
        """Preload hot experts to GPU cache."""
        if plan.hot_expert_slots == 0:
            return

        # Load first N experts (greedy allocation)
        experts_loaded = 0
        target = min(plan.hot_expert_slots, plan.model_info.num_layers * plan.model_info.num_experts)

        for layer_id in range(plan.model_info.num_layers):
            for expert_id in range(plan.model_info.num_experts):
                if experts_loaded >= target:
                    break

                # Load expert weights
                expert_weights = self._load_expert_from_storage(model_path, layer_id, expert_id)

                # Preload to GPU cache
                expert_cache.preload_gpu(layer_id, expert_id, expert_weights)

                experts_loaded += 1

            if experts_loaded >= target:
                break

        log(f"   ✓ Preloaded {experts_loaded} hot experts to GPU")

    def _load_expert_from_storage(
        self,
        model_path: Path,
        layer_id: int,
        expert_id: int
    ) -> dict[str, torch.Tensor]:
        """Load specific expert weights from storage."""
        expert_weights = {}

        # Expert weight patterns (universal across architectures)
        patterns = [
            f"model.layers.{layer_id}.block_sparse_moe.experts.{expert_id}.*.weight",
            f"model.layers.{layer_id}.mlp.experts.{expert_id}.*.weight",
            f"model.layers.{layer_id}.moe.experts.{expert_id}.*.weight"
        ]

        # Load from safetensors files
        safetensors_files = list(model_path.glob("*.safetensors"))

        for st_file in safetensors_files:
            weights = load_file(st_file, device="cpu")

            for key, tensor in weights.items():
                # Check if matches expert pattern
                for pattern in patterns:
                    if self._matches_pattern(key, pattern):
                        expert_weights[key] = tensor

        if not expert_weights:
            raise ValueError(f"No weights found for expert ({layer_id}, {expert_id})")

        return expert_weights

    def _is_shared_weight(self, weight_name: str) -> bool:
        """Check if weight is shared (non-expert)."""
        # Use existing SharedWeightPlacer logic
        # Shared if NOT expert
        return not self._is_expert_weight(weight_name)

    def _is_expert_weight(self, weight_name: str) -> bool:
        """Check if weight is expert weight."""
        # Expert weights have "experts" in the path
        return ".experts." in weight_name or ".expert." in weight_name

    def _matches_pattern(self, text: str, pattern: str) -> bool:
        """Match glob-like pattern."""
        regex_pattern = pattern.replace(".", r"\.").replace("*", ".*")
        return bool(re.match(regex_pattern, text))

    def _total_bytes(self, weights: dict[str, torch.Tensor]) -> int:
        """Calculate total bytes of weight dict."""
        return sum(t.numel() * t.element_size() for t in weights.values())

    def _format_bytes(self, bytes: int) -> str:
        """Format bytes as human-readable string."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes < 1024:
                return f"{bytes:.1f}{unit}"
            bytes /= 1024
        return f"{bytes:.1f}TB"
