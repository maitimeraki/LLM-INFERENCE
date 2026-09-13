"""
Custom PyTorch MoE Inference Engine

Main orchestrator for Mixture-of-Experts inference with intelligent prefill/decode
separation and expert caching. Achieves 20-30 tokens/sec decode speed through:
- Batch router computation during prefill
- Expert preloading based on frequency analysis
- LRU expert cache with predictive prefetching
- Efficient KV cache management

Target: 20-30 tok/s decode speed on consumer hardware (RTX 3090/4090)
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Dict, List, Tuple, Iterator, Callable
from dataclasses import dataclass
from collections import Counter

import torch
import torch.nn as nn
from pathlib import Path
from safetensors import safe_open

from sparse_llm.inference.router_calculator import RouterCalculator
from sparse_llm.inference.expert_processor import ExpertProcessor, ExpertFFN
from sparse_llm.inference.attention_engine import AttentionEngine, RMSNorm
from sparse_llm.inference.expert_cache_manager import ExpertCacheManager
try:
    from sparse_llm.core.router_predictor import RouterPredictor
except ImportError:
    from sparse_llm.core.router import RouterPredictor
from sparse_llm.loading.model_introspector import ModelIntrospector, ModelInfo
from sparse_llm.loading.memory_budget_calculator import (
    DynamicMemoryBudgetCalculator,
    UserRequest,
    MemoryAllocation
)
from sparse_llm.loading.resource_budget import ResourceBudget

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for inference engine."""
    # Model configuration
    model_path: str
    device: str = "cuda"
    dtype: str = "float16"

    # Expert cache configuration
    expert_cache_size: int = 4  # Number of experts to keep in GPU
    enable_predictive_prefetch: bool = True
    prefetch_confidence_threshold: float = 0.30
    prefetch_top_k: int = 15

    # Generation parameters (defaults)
    max_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50

    # Memory configuration
    max_seq_len: int = 4096
    allow_cpu_offload: bool = True
    allow_ssd_offload: bool = True

    # Expert processing mode
    expert_processing_mode: str = "standard"  # "standard" | "tiled" | "fused"

    # Performance tuning
    use_flash_attention: bool = True
    compile_model: bool = False  # torch.compile support


@dataclass
class GenerationState:
    """Internal state during generation."""
    prompt_tokens: torch.Tensor
    kv_cache: Dict[str, torch.Tensor]
    hidden_states: torch.Tensor
    generated_tokens: List[int]
    token_times: List[float]  # Per-token generation time
    cache_hit_count: int = 0
    cache_miss_count: int = 0


@dataclass
class GenerationStats:
    """Statistics from generation."""
    total_tokens: int
    prompt_tokens: int
    generated_tokens: int
    total_time: float
    prefill_time: float
    decode_time: float
    tokens_per_second: float
    cache_hit_rate: float
    memory_usage_gb: float
    per_token_times: List[float]


class CustomMoEInferenceEngine:
    """
    Main inference engine for custom PyTorch MoE implementation.

    Architecture:
    - Prefill phase: Batch process prompt, analyze expert usage, preload hot experts
    - Decode phase: Single-token generation with expert caching and prefetching
    - Cleanup: Release resources between requests

    Performance targets:
    - Prefill: 200-500 tokens/sec throughput
    - Decode: 20-30 tokens/sec with 80%+ cache hit rate
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[InferenceConfig] = None
    ):
        """
        Initialize the inference engine.

        Args:
            model_path: Path to model weights or Hugging Face model ID
            config: Optional configuration override
        """
        self.model_path = model_path
        self.config = config or InferenceConfig(model_path=model_path)

        # Device setup
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        self.dtype = self._get_torch_dtype(self.config.dtype)

        logger.info(f"Initializing CustomMoEInferenceEngine on {self.device}")
        logger.info(f"Model: {model_path}, dtype: {self.config.dtype}")

        # Introspect model architecture
        self.introspector = ModelIntrospector()
        self.model_info = self._load_model_info()

        if not self.model_info.is_moe:
            logger.warning("Model is not MoE - this engine is optimized for MoE models")

        # Calculate memory budget
        self.memory_allocation = self._calculate_memory_budget()

        # Initialize components
        self.tokenizer = self._load_tokenizer()
        self.router_calculator = self._initialize_router()
        self.attention_engine = self._initialize_attention()
        self.expert_cache = self._initialize_expert_cache()
        self.cache_manager = self._initialize_cache_manager()
        self.expert_processor = self._initialize_expert_processor()

        # Safetensors key→file index (built once after model path is available)
        self._safetensors_index: dict[str, tuple[Path, str]] = {}
        # Module cache: (layer_id, expert_id) → ExpertFFN, built once per (layer, expert)
        self._expert_module_cache: dict[tuple[int, int], ExpertFFN] = {}

        # Load model weights
        self._load_model_weights()

        # Statistics tracking
        self.total_generated_tokens = 0
        self.total_generation_time = 0.0
        self.total_requests = 0

        logger.info("Engine initialized successfully")
        self._log_configuration()

    def _get_torch_dtype(self, dtype: str) -> torch.dtype:
        """Convert string dtype to torch dtype."""
        dtype_map = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        return dtype_map.get(dtype.lower(), torch.float16)

    def _load_model_info(self) -> ModelInfo:
        """Load and introspect model architecture."""
        logger.info("Introspecting model architecture...")
        model_info = self.introspector.introspect(self.model_path)

        logger.info(f"Model info: {model_info.num_layers} layers, "
                   f"{model_info.num_experts} experts, "
                   f"experts per token: {model_info.num_experts_per_tok}")
        logger.info(f"Shared weights: {model_info.shared_weight_bytes / 1024**3:.2f}GB, "
                   f"Expert weights: {model_info.expert_weight_bytes / 1024**3:.2f}GB each")

        return model_info

    def _calculate_memory_budget(self) -> MemoryAllocation:
        """Calculate memory allocation plan using three-tier system."""
        from sparse_llm.loading.resource_budget import ResourceBudget

        # Get available resources (Phase 1)
        resource_budget = ResourceBudget.from_system()

        # Create user request
        user_request = UserRequest(
            max_model_len=self.config.max_seq_len,
            dtype=self.config.dtype,
            quantization=None,
            tensor_parallel_size=1,
            allow_cpu_offload=self.config.allow_cpu_offload,
            allow_ssd_offload=self.config.allow_ssd_offload
        )

        # Calculate allocation with three-tier distribution
        calculator = DynamicMemoryBudgetCalculator()
        allocation = calculator.calculate(resource_budget, self.model_info, user_request)

        if not allocation.can_fulfill:
            raise RuntimeError(f"Cannot fulfill memory requirements: {allocation.rejection_reason}")

        logger.info(f"Three-tier memory allocation plan:")
        logger.info(f"  GPU Tier (Hot): {allocation.gpu_hot_expert_count} experts, "
                   f"{allocation.gpu_hot_experts / 1024**3:.2f}GB")
        logger.info(f"  CPU Tier (Warm): {allocation.cpu_warm_expert_count} experts, "
                   f"{allocation.cpu_warm_experts / 1024**3:.2f}GB")
        logger.info(f"  Storage Tier (Cold): {allocation.ssd_cold_expert_count} experts, "
                   f"{allocation.ssd_cold_experts / 1024**3:.2f}GB")
        logger.info(f"  Total GPU: {allocation.total_gpu_required / 1024**3:.2f}GB / "
                   f"{allocation.total_gpu_available / 1024**3:.2f}GB available")

        return allocation

    def _load_tokenizer(self):
        """Load tokenizer for the model."""
        from transformers import AutoTokenizer

        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=False)

        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def _initialize_router(self) -> RouterCalculator:
        """Initialize router calculator with real router weights.

        Router weights will be loaded from the model checkpoint during _load_model_weights().
        This just creates the calculator object - weights are set later.
        """
        logger.info("Initializing router calculator (weights will be loaded from checkpoint)...")

        # Create router without weights initially - will be set during weight loading
        router = RouterCalculator(
            router_weights=None,  # Will be set during _load_model_weights
            num_experts=self.model_info.num_experts,
            top_k=self.model_info.num_experts_per_tok,
            hidden_dim=self.model_info.hidden_size  # Provide hidden_dim for lazy init
        )

        return router

    def _initialize_attention(self) -> AttentionEngine:
        """Initialize attention engine with KV cache support."""
        logger.info("Initializing attention engine...")

        head_dim = self.model_info.hidden_size // self.model_info.num_attention_heads

        attention = AttentionEngine(
            num_layers=self.model_info.num_layers,
            num_heads=self.model_info.num_attention_heads,
            head_dim=head_dim,
            max_seq_len=self.config.max_seq_len,
            device=str(self.device),
            dtype=self.dtype,
            # GQA/MQA: ModelInfo exposes num_key_value_heads (defaults to
            # num_attention_heads for standard MHA). getattr keeps bare/legacy
            # engines without the field on the MHA path.
            num_kv_heads=getattr(self.model_info, "num_key_value_heads", None),
        )

        return attention

    def _initialize_expert_cache(self) -> ExpertCache:
        """Initialize three-tier expert cache (GPU → CPU → Storage)."""
        logger.info("Initializing three-tier expert cache...")

        # Import the three-tier cache from loading module
        from sparse_llm.loading.expert_cache import ExpertCache as ThreeTierCache

        # Use memory allocation tier counts
        gpu_slots = self.memory_allocation.gpu_hot_expert_count
        cpu_slots = self.memory_allocation.cpu_warm_expert_count
        expert_bytes = self.model_info.expert_weight_bytes

        logger.info(f"Three-tier cache configuration:")
        logger.info(f"  GPU slots (hot): {gpu_slots}")
        logger.info(f"  CPU slots (warm): {cpu_slots}")
        logger.info(f"  Storage (cold): {self.memory_allocation.ssd_cold_expert_count} experts")

        # Storage loader will be set later in _load_model_weights
        cache = ThreeTierCache(
            gpu_slots=gpu_slots,
            cpu_slots=cpu_slots,
            expert_bytes=expert_bytes,
            storage_loader=None  # Will be set after model path is available
        )

        return cache

    def _initialize_cache_manager(self) -> ExpertCacheManager:
        """Initialize cache manager with predictive prefetching."""
        logger.info("Initializing cache manager...")

        # Create router predictor if enabled
        predictor = None
        if self.config.enable_predictive_prefetch:
            predictor = RouterPredictor(
                num_experts=self.model_info.num_experts,
                sequence_length=10  # Track last 10 expert selections
            )

        manager = ExpertCacheManager(
            expert_cache=self.expert_cache,
            router_predictor=predictor,
            enable_prefetch=self.config.enable_predictive_prefetch,
            prefetch_confidence_threshold=self.config.prefetch_confidence_threshold,
            prefetch_top_k=self.config.prefetch_top_k
        )

        return manager

    def _initialize_expert_processor(self) -> ExpertProcessor:
        """Initialize expert processor."""
        logger.info("Initializing expert processor...")

        processor = ExpertProcessor(
            expert_cache=self.cache_manager,
            num_experts=self.model_info.num_experts,
            hidden_dim=self.model_info.hidden_size,
            expert_dim=self.model_info.intermediate_size,
            activation="gelu",
            device=str(self.device),
            enable_predictive_prefetch=self.config.enable_predictive_prefetch,
            processing_mode=getattr(self.config, 'expert_processing_mode', 'standard'),
        )

        return processor

    def _load_model_weights(self):
        """Load model weights using three-tier placement strategy."""
        logger.info("Loading model weights with three-tier placement...")

        from pathlib import Path
        from safetensors import safe_open
        from huggingface_hub import snapshot_download

        # Step 1: Ensure model is downloaded
        model_path = self._ensure_model_downloaded()

        # Step 2: Load shared weights to GPU
        logger.info(f"[1/4] Loading shared weights to GPU...")
        self.shared_weights = self._load_shared_weights(model_path)

        # Step 3: NO expert preloading - memory constrained system
        # Experts will be loaded on-demand from storage
        logger.info(f"[2/4] Expert preloading disabled (memory constrained)")
        logger.info(f"    Experts will be loaded on-demand from storage")

        # Step 4: Configure storage loader for on-demand expert loading
        logger.info(f"[3/4] Configuring expert storage loader (on-demand)...")

        def on_demand_expert_loader(layer_id: int, expert_id: int):
            """Load expert from storage on-demand using subprocess."""
            key = (layer_id, expert_id)
            if key in self._expert_weights_cpu:
                return self._expert_weights_cpu[key]
            # Try subprocess loading
            try:
                weights = self._load_expert_single_in_subprocess(
                    self.model_path, layer_id, expert_id
                )
                if weights:
                    self._expert_weights_cpu[key] = weights
                    return weights
            except Exception as e:
                logger.warning(f"Failed to load expert {key}: {e}")
            return None

        self.expert_cache.storage_loader = on_demand_expert_loader

        # Step 5: No hot experts preloaded due to memory constraints
        logger.info(f"[4/4] No hot experts preloaded (on-demand loading only)")

        # Step 5: Log tier distribution
        logger.info(f"[4/4] Weight loading complete:")
        logger.info(f"  ✓ Shared weights: {len(self.shared_weights)} tensors on GPU")
        logger.info(f"  ✓ Hot experts (GPU): {self.memory_allocation.gpu_hot_expert_count}")
        logger.info(f"  ✓ Warm experts (CPU): {self.memory_allocation.cpu_warm_expert_count}")
        logger.info(f"  ✓ Cold experts (Storage): {self.memory_allocation.ssd_cold_expert_count}")

        # Log actual memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    def _ensure_model_downloaded(self) -> Path:
        """Ensure model is available locally."""
        from pathlib import Path
        from huggingface_hub import snapshot_download

        local_path = Path(self.model_path)
        if local_path.exists():
            logger.info(f"Using local model: {local_path}")
            return local_path

        logger.info(f"Downloading model from HuggingFace: {self.model_path}")
        downloaded_path = snapshot_download(
            self.model_path,
            allow_patterns=["*.safetensors", "*.json"],
            ignore_patterns=["*.bin"]  # Prefer safetensors over pickle
        )
        return Path(downloaded_path)

    def _load_shared_weights(self, model_path: Path) -> dict[str, torch.Tensor]:
        """Load shared weights (embeddings, attention, norms, router) to GPU.

        Uses subprocess to avoid safetensors mmap exhaustion on Windows.
        """
        shared_weights = {}

        # Find safetensors files
        safetensors_files = list(model_path.glob("*.safetensors"))
        if not safetensors_files:
            safetensors_files = list(model_path.glob("**/*.safetensors"))

        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors files found in {model_path}")

        # Build index by loading file metadata in subprocess (not full tensors)
        logger.info("   Building safetensors index...")
        self._build_safetensors_index(model_path)

        # Group shared weights by file
        shared_keys_by_file: dict[Path, list[str]] = {}
        for key, (st_file, _) in self._safetensors_index.items():
            if self._is_shared_weight(key):
                shared_keys_by_file.setdefault(st_file, []).append(key)

        total_bytes = 0
        router_weights_by_layer = {}

        # Load each file's shared weights in subprocess
        for st_file, keys in shared_keys_by_file.items():
            logger.info(f"   Loading {len(keys)} shared weights from {st_file.name}...")
            try:
                weights = self._load_file_in_subprocess(st_file)
                for key in keys:
                    if key in weights:
                        tensor = weights[key].to(device=self.device, dtype=self.dtype)
                        shared_weights[key] = tensor
                        total_bytes += tensor.numel() * tensor.element_size()

                        if ".gate.weight" in key and "layers" in key and "experts" not in key:
                            try:
                                layer_num = int(key.split(".layers.")[1].split(".")[0])
                                router_weights_by_layer[layer_num] = tensor
                            except (IndexError, ValueError):
                                pass
            except Exception as e:
                logger.warning(f"   Failed to load shared weights from {st_file.name}: {e}")

        logger.info(f"   Loaded {len(shared_weights)} shared weights ({total_bytes / 1024**3:.2f}GB)")

        # Look for router gate weights
        router_key = None
        for key in shared_weights.keys():
            if ".block_sparse_moe.gate.weight" in key and "layers" in key:
                router_key = key
                break
            if ".mlp.gate.weight" in key and "layers" in key:
                router_key = key
                break

        # Load router weights from checkpoint or fall back to random
        if router_key and router_key in shared_weights:
            real_weights = shared_weights[router_key].to(device=self.device, dtype=self.dtype)
            self.router_calculator.set_router_weights(real_weights)
            logger.info(f"   ✓ Real router weights: {router_key}, shape {real_weights.shape}")
        else:
            logger.warning("   ⚠ Router weights not found, using random")
            random_weights = torch.randn(
                self.model_info.hidden_size,
                self.model_info.num_experts,
                device=self.device,
                dtype=self.dtype
            )
            self.router_calculator.set_router_weights(random_weights)
        self.router_weights_by_layer = {}

        self._load_attention_and_norm_weights(shared_weights)

        return shared_weights

    def _load_attention_and_norm_weights(self, shared_weights: dict) -> None:
        """Eager weight-load timing: bind every layer's attention projections and
        pre/post norms now, so missing keys are reported at load time."""
        hidden = self.model_info.hidden_size
        self.layer_norms: dict = {}
        self._warned_norms: set = set()

        for layer_id in range(self.model_info.num_layers):
            self.attention_engine.load_layer_weights(shared_weights, layer_id)
            self.layer_norms[layer_id] = {
                "input_layernorm": self._load_norm(
                    shared_weights, layer_id, "input_layernorm", hidden
                ),
                "post_attention_layernorm": self._load_norm(
                    shared_weights, layer_id, "post_attention_layernorm", hidden
                ),
            }

        self.final_norm = self._load_norm(shared_weights, None, "norm", hidden)

    def _build_safetensors_index(self, model_path: Path) -> None:
        """Build key→(file, key) index for all safetensors files once at init.

        Uses comprehensive subprocess worker to avoid safetensors mmap exhaustion.
        """
        import pickle
        import subprocess
        import tempfile
        import os

        files = list(model_path.glob("*.safetensors"))
        if not files:
            files = list(model_path.glob("**/*.safetensors"))

        # Use comprehensive worker that loads all weights in one subprocess
        worker_script = Path(__file__).parent.parent.parent / "load_all_worker.py"
        output_path = tempfile.mktemp(suffix='.pkl')

        try:
            logger.info(f"   Loading all weights via comprehensive worker...")
            result = subprocess.run(
                ['python', str(worker_script), str(model_path), output_path],
                capture_output=True,
                text=True,
                timeout=600  # 10 min timeout
            )

            if result.returncode == 0 and os.path.exists(output_path):
                with open(output_path, 'rb') as f:
                    data = pickle.load(f)

                # Build index from loaded data
                self._safetensors_index = {}
                for key in data.get('shared', {}).keys():
                    # Find which file this key is in
                    for st_file in files:
                        # We don't have file mapping anymore, so store as is
                        self._safetensors_index[key] = (st_file, key)

                logger.info(f"   Loaded {data.get('num_shared', 0)} shared weights, {data.get('num_experts', 0)} experts")
                return
            else:
                logger.warning(f"   Worker failed: {result.stderr[:500] if result.stderr else 'unknown error'}")

        except subprocess.TimeoutExpired:
            logger.warning("   Worker timed out")
        except Exception as e:
            logger.warning(f"   Worker exception: {e}")
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

        # Fallback: empty index
        self._safetensors_index = {}
        logger.warning("   Using empty safetensors index - weights will be loaded on-demand")

    def _load_norm(self, shared_weights: dict, layer_id, name: str, hidden: int):
        """Build an RMSNorm from checkpoint weights, or None (identity fallback)."""
        if layer_id is None:
            keys = ("model.norm.weight", "norm.weight")
        else:
            keys = (
                f"model.layers.{layer_id}.{name}.weight",
                f"layers.{layer_id}.{name}.weight",
            )
        for key in keys:
            if key in shared_weights:
                norm = RMSNorm(hidden, eps=1e-6).to(device=self.device, dtype=self.dtype)
                with torch.no_grad():
                    norm.weight.copy_(shared_weights[key].to(device=self.device, dtype=self.dtype))
                return norm
        if name not in self._warned_norms:
            self._warned_norms.add(name)
            logger.warning(f"Missing norm weights for '{name}'; using identity fallback.")
        return None

    @staticmethod
    def _as_norm(norm):
        """Accept an nn.Module, a raw weight Tensor (tests), or None."""
        if norm is None:
            return None
        if isinstance(norm, torch.Tensor):
            mod = RMSNorm(norm.shape[-1])
            with torch.no_grad():
                mod.weight.copy_(norm)
            return mod
        return norm

    def _get_norm(self, layer_idx: int, name: str):
        """Per-layer norm module; identity when absent."""
        entry = getattr(self, "layer_norms", {}).get(layer_idx)
        if not isinstance(entry, dict):
            return torch.nn.Identity()
        norm = self._as_norm(entry.get(name))
        if norm is None:
            return torch.nn.Identity()
        # Cache a wrapped module so test-injected raw tensors wrap only once.
        entry[name] = norm
        if hasattr(norm, "to"):
            norm = norm.to(self.device)
            entry[name] = norm
        return norm

    def _apply_final_norm(self, h: torch.Tensor) -> torch.Tensor:
        norm = self._as_norm(getattr(self, "final_norm", None))
        return h if norm is None else norm(h)

    def _apply_attention_layer(
        self,
        h: torch.Tensor,
        layer_idx: int,
        kv_cache,
        mode: str = "decode",
    ):
        """One attention sublayer with its residual: h = h + attn(norm(h)).

        `mode="prefill"` runs the whole prompt through one layer; `"decode"`
        processes a single token. The caller owns the per-layer loop.
        """
        norm = self._get_norm(layer_idx, "input_layernorm")
        normed = norm(h)

        if mode == "prefill":
            attn_out, kv_cache = self.attention_engine.prefill(
                input_ids=normed, layer_idx=layer_idx, kv_cache=kv_cache
            )
        else:
            attn_out, kv_cache = self.attention_engine.decode(
                token_id=normed, layer_idx=layer_idx, kv_cache=kv_cache
            )

        return h + attn_out, kv_cache

    def _apply_moe_layer(self, h: torch.Tensor, layer_idx: int, batch_mode: bool) -> torch.Tensor:
        """One MoE sublayer with its residual: h = h + moe(norm(h))."""
        norm = self._get_norm(layer_idx, "post_attention_layernorm")
        normed = norm(h)

        # Use layer-specific router weights if available.
        if hasattr(self, 'router_weights_by_layer') and layer_idx in self.router_weights_by_layer:
            original_weights = self.router_calculator.router_weights
            self.router_calculator.router_weights = self.router_weights_by_layer[layer_idx]
            expert_indices, expert_weights = self.router_calculator.forward(
                normed, batch_mode=batch_mode
            )
            self.router_calculator.router_weights = original_weights
        else:
            expert_indices, expert_weights = self.router_calculator.forward(
                normed, batch_mode=batch_mode
            )

        if batch_mode:
            activated_experts = self._analyze_expert_frequency(expert_indices)
            if layer_idx == 0:  # Only log for first layer to avoid spam
                logger.info(f"Layer {layer_idx} prefill activated {len(activated_experts)} "
                            f"unique experts: {activated_experts[:10]}")
            # Dispatch based on processing mode
            mode = getattr(self.expert_processor, 'processing_mode', 'standard')

            if mode == "fused":
                out = self.expert_processor.process_batch_fused(
                    hidden_states=normed,
                    expert_indices=expert_indices,
                    expert_weights=expert_weights,
                    layer_id=layer_idx,
                    expert_loader=self._create_expert_loader(layer_idx),
                    trigger_prefetch=(layer_idx == self.model_info.num_layers - 1),
                )
            elif mode == "tiled":
                out = self.expert_processor.process_batch_tiled(
                    hidden_states=normed,
                    expert_indices=expert_indices,
                    expert_weights=expert_weights,
                    layer_id=layer_idx,
                    expert_loader=self._create_expert_loader(layer_idx),
                    trigger_prefetch=(layer_idx == self.model_info.num_layers - 1),
                )
            else:  # standard
                out = self.expert_processor.process_batch(
                    hidden_states=normed,
                    expert_indices=expert_indices,
                    expert_weights=expert_weights,
                    layer_id=layer_idx,
                    expert_loader=self._create_expert_loader(layer_idx),
                    trigger_prefetch=(layer_idx == self.model_info.num_layers - 1),
                )
            # Learn routing patterns for prefetching
            self.expert_processor.observe_routing(activated_experts, layer_id=layer_idx)
            # Prefetch next layer's predicted experts
            if layer_idx < self.model_info.num_layers - 1:
                self.expert_processor.prefetch_next(activated_experts, layer_id=layer_idx + 1)
        else:
            activated_experts = self._analyze_expert_frequency(expert_indices)
            out = self.expert_processor.process_single(
                hidden_state=normed,
                expert_indices=expert_indices.squeeze(1),  # [batch, top_k]
                expert_weights=expert_weights.squeeze(1),
                layer_id=layer_idx,
                expert_loader=self._create_expert_loader(layer_idx),
                update_predictor=True,
            )
            # Observe routing for decode mode
            self.expert_processor.observe_routing(activated_experts, layer_id=layer_idx)

        return h + out

    def _forward_one_token(self, current_hidden: torch.Tensor, kv_cache):
        """Run every layer for one decode token. Shared by both decode paths.

        R7 seq_len contract: AttentionEngine.decode writes K/V at
        `kv_cache['seq_len']` and does NOT advance it. We advance it exactly
        once per generated token, here, after the full layer loop. Advancing
        inside `decode` would overshoot by num_layers per token.
        """
        for layer_id in range(self.model_info.num_layers):
            current_hidden, kv_cache = self._apply_attention_layer(
                current_hidden, layer_id, kv_cache, mode="decode"
            )
            current_hidden = self._apply_moe_layer(current_hidden, layer_id, batch_mode=False)

        kv_cache['seq_len'] = kv_cache.get('seq_len', 0) + 1
        return self._apply_final_norm(current_hidden), kv_cache

    def _preload_all_experts_to_cpu(self, model_path: Path) -> None:
        """Preload expert weights into CPU RAM using subprocess to avoid mmap exhaustion.

        Root cause: safetensors.torch.load_file() accumulates mmap resources on Windows
        that eventually cause SIGSEGV after multiple large file loads. The fix is to load
        each large file in a fresh subprocess, extract tensors, and return via pickle.

        ponytail: Only load first 5 large files to avoid memory exhaustion.
        Missing experts (layers 17-23) will be handled with on-demand loading.
        """
        import gc
        import re
        total_experts = self.model_info.num_layers * self.model_info.num_experts

        files = list(model_path.glob("*.safetensors"))
        if not files:
            files = list(model_path.glob("**/*.safetensors"))

        # Sort files by size to load smallest first
        files = sorted(files, key=lambda f: f.stat().st_size)

        # Only process the first 5 files to avoid memory exhaustion
        max_files = 5
        files_to_load = files[:max_files]
        skipped_files = files[max_files:]

        if skipped_files:
            logger.warning(f"   Skipping {len(skipped_files)} large files to avoid memory exhaustion")

        loaded = 0
        for st_file in files_to_load:
            file_size_mb = st_file.stat().st_size / 1024**2
            logger.info(f"   Loading {st_file.name} ({file_size_mb:.0f}MB)...")

            try:
                weights = self._load_file_in_subprocess(st_file)
                if weights:
                    self._expert_weights_cpu.update(weights)
                    loaded += len(weights)
                    logger.info(f"   ✓ {st_file.name}: {len(weights)} experts loaded")
                gc.collect()
            except Exception as e:
                logger.warning(f"   Failed to load {st_file.name}: {e}")

        logger.info(f"   Loaded {loaded}/{total_experts} experts to CPU RAM")

        if self._expert_weights_cpu:
            sample = next(iter(self._expert_weights_cpu.values()))
            per_expert_bytes = sum(t.numel() * t.element_size() for t in sample.values())
            total_bytes = per_expert_bytes * len(self._expert_weights_cpu)
            logger.info(f"   Expert weights RAM: {total_bytes / 1024**3:.2f}GB")

    def _extract_experts_from_file(self, st_file: Path) -> dict:
        """Extract all expert weights from a safetensors file."""
        import re
        import safetensors.torch

        result = {}
        all_tensors = safetensors.torch.load_file(str(st_file))

        pattern = re.compile(
            r"model\.layers\.(\d+)\.(?:block_sparse_moe|mlp|moe|feed_forward)\.experts\.(\d+)\."
        )

        expert_keys: dict[tuple[int, int], list[str]] = {}
        for key_name in all_tensors.keys():
            m = pattern.match(key_name)
            if m:
                layer_id = int(m.group(1))
                expert_id = int(m.group(2))
                expert_keys.setdefault((layer_id, expert_id), []).append(key_name)

        for (layer_id, expert_id), keys in expert_keys.items():
            cache_key = (layer_id, expert_id)
            expert_weights = {}
            for key_name in keys:
                tensor = all_tensors[key_name]
                suffix = key_name.split(f".experts.{expert_id}.")[-1]
                if "gate_proj.weight" in suffix:
                    expert_weights["w1.weight"] = tensor
                elif "down_proj.weight" in suffix:
                    expert_weights["w2.weight"] = tensor
                elif "up_proj.weight" in suffix:
                    expert_weights["w3.weight"] = tensor

            required_keys = {"w1.weight", "w2.weight"}
            if required_keys.issubset(expert_weights.keys()):
                result[cache_key] = expert_weights

        return result

    def _load_expert_single_in_subprocess(self, model_path, layer_id, expert_id) -> Optional[dict]:
        """Load a single expert using subprocess."""
        import subprocess
        import tempfile
        import os

        worker_script = Path(__file__).parent.parent.parent / "load_expert_worker.py"
        if not worker_script.exists():
            return None

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            output_path = tmp.name

        try:
            result = subprocess.run(
                ['python', str(worker_script), str(model_path), str(layer_id), str(expert_id), output_path],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0 and os.path.exists(output_path):
                import pickle
                with open(output_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Expert load subprocess failed: {e}")
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
        return None

    def _load_file_in_subprocess(self, st_file: Path) -> dict:
        """Load expert weights from a safetensors file in a subprocess."""
        import pickle
        import subprocess
        import tempfile
        import os

        # Use the standalone worker script
        worker_script = Path(__file__).parent.parent.parent / "load_experts_worker.py"
        if not worker_script.exists():
            print(f"DEBUG: Worker script not found at {worker_script}", flush=True)
            return {}

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            output_path = tmp.name

        try:
            # Run worker script as separate process
            print(f"DEBUG: Running: python {worker_script} {st_file} {output_path}", flush=True)
            result = subprocess.run(
                ['python', str(worker_script), str(st_file), output_path],
                capture_output=True,
                text=True,
                timeout=300  # 5 min timeout
            )
            print(f"DEBUG: Worker stdout: {result.stdout[:200] if result.stdout else 'empty'}", flush=True)
            print(f"DEBUG: Worker stderr: {result.stderr[:200] if result.stderr else 'empty'}", flush=True)
            if result.returncode != 0:
                return {}

            # Load the pickled result
            with open(output_path, 'rb') as f:
                weights = pickle.load(f)

            return weights
        except subprocess.TimeoutExpired:
            print(f"DEBUG: Worker timed out for {st_file.name}", flush=True)
            return {}
        except Exception as e:
            print(f"DEBUG: Worker exception: {e}", flush=True)
            return {}
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def _load_expert_from_safetensors(
        self,
        model_path: Path,
        layer_id: int,
        expert_id: int
    ) -> Optional[dict[str, torch.Tensor]]:
        """Load specific expert weights from safetensors files.

        Returns None if expert not found.
        """
        expert_weights = {}

        # Build candidate key prefixes for this (layer, expert)
        prefix_patterns = [
            f"model.layers.{layer_id}.block_sparse_moe.experts.{expert_id}.",
            f"model.layers.{layer_id}.mlp.experts.{expert_id}.",
            f"model.layers.{layer_id}.moe.experts.{expert_id}.",
            f"model.layers.{layer_id}.feed_forward.experts.{expert_id}.",
        ]

        # Group matching keys by file to batch reads
        keys_by_file: dict[Path, list[str]] = {}
        for key, (st_file, _canonical) in self._safetensors_index.items():
            for prefix in prefix_patterns:
                if key.startswith(prefix):
                    keys_by_file.setdefault(st_file, []).append(key)
                    break

        if not keys_by_file:
            return None

        # Open each file once, read all needed tensors, close immediately
        raw_weights: dict[str, torch.Tensor] = {}
        for st_file, keys in keys_by_file.items():
            # Load ALL tensors from file at once, then extract what we need.
            # This avoids repeated open/close of large files.
            import safetensors.torch
            print(f"DEBUG: Loading {st_file.name}...", flush=True)
            all_tensors = safetensors.torch.load_file(str(st_file))
            print(f"DEBUG: Loaded {st_file.name}, {len(all_tensors)} tensors", flush=True)
            for key in keys:
                if key in all_tensors:
                    raw_weights[key] = all_tensors[key]
            # Clear reference to allow GC
            del all_tensors

        if not raw_weights:
            return None

        # Normalize to w1/w2/w3
        for key, tensor in raw_weights.items():
            parts = key.split(f".experts.{expert_id}.")
            if len(parts) == 2:
                weight_name = parts[1]
            else:
                weight_name = key.split(f".{expert_id}.")[-1]

            if "w1.weight" in weight_name or "gate_proj.weight" in weight_name:
                expert_weights["w1.weight"] = tensor
            elif "w2.weight" in weight_name or "down_proj.weight" in weight_name:
                expert_weights["w2.weight"] = tensor
            elif "w3.weight" in weight_name or "up_proj.weight" in weight_name:
                expert_weights["w3.weight"] = tensor
            else:
                expert_weights[weight_name] = tensor

        required_keys = {"w1.weight", "w2.weight"}
        if not required_keys.issubset(expert_weights.keys()):
            logger.warning(f"Expert ({layer_id}, {expert_id}) missing weights: {list(expert_weights.keys())}")
            return None

        return expert_weights

    def _load_expert_from_storage(
        self,
        model_path: Path,
        layer_id: int,
        expert_id: int
    ) -> dict[str, torch.Tensor]:
        """Load specific expert weights from CPU RAM cache."""
        key = (layer_id, expert_id)
        if key not in self._expert_weights_cpu:
            raise FileNotFoundError(
                f"Expert ({layer_id}, {expert_id}) not found in CPU RAM cache. "
                f"All experts should be preloaded during initialization."
            )
        return self._expert_weights_cpu[key]

    def _preload_hot_experts(self, model_path: Path) -> None:
        """Preload hot experts: build ExpertFFN modules and populate the module cache.

        Uses pre-loaded CPU RAM weights (already in self._expert_weights_cpu) and
        moves them to GPU. This avoids any safetensors access during inference.
        """
        hot_count = self.memory_allocation.gpu_hot_expert_count
        total_experts = self.model_info.num_layers * self.model_info.num_experts
        target = min(hot_count, total_experts)

        experts_loaded = 0
        for layer_id in range(self.model_info.num_layers):
            for expert_id in range(self.model_info.num_experts):
                if experts_loaded >= target:
                    break

                key = (layer_id, expert_id)
                if key in self._expert_module_cache:
                    experts_loaded += 1
                    continue

                # Get weights from pre-loaded CPU RAM (no safetensors access)
                expert_weights = self._expert_weights_cpu.get(key)
                if expert_weights is None:
                    logger.warning(f"Expert {key} not in CPU RAM cache")
                    continue

                has_gate = "w3.weight" in expert_weights

                expert = ExpertFFN(
                    hidden_dim=self.model_info.hidden_size,
                    expert_dim=self.model_info.intermediate_size,
                    activation="silu",
                    has_gate=has_gate,
                )
                expert.w1.weight.data = expert_weights["w1.weight"].to(
                    device=self.device, dtype=self.dtype
                )
                expert.w2.weight.data = expert_weights["w2.weight"].to(
                    device=self.device, dtype=self.dtype
                )
                if has_gate and expert.w3 is not None:
                    expert.w3.weight.data = expert_weights["w3.weight"].to(
                        device=self.device, dtype=self.dtype
                    )
                expert.eval()

                self._expert_module_cache[key] = expert
                # Also populate three-tier cache tensors so get() has GPU tensors
                self.expert_cache.preload_gpu(layer_id, expert_id, expert_weights)

                experts_loaded += 1

            if experts_loaded >= target:
                break

        logger.info(f"   ✓ Preloaded {experts_loaded}/{target} hot ExpertFFN modules")

    def _is_shared_weight(self, weight_name: str) -> bool:
        """Check if weight is shared (non-expert)."""
        # Shared weights: embeddings, attention, norms, router, lm_head
        shared_patterns = [
            "embed_tokens",
            "wte",
            "self_attn",
            "layernorm",
            "input_layernorm",
            "post_attention_layernorm",
            "ln",
            "norm",
            "gate",  # Router gate (not expert gate_proj)
            "lm_head"
        ]

        # Expert patterns to exclude
        expert_patterns = [
            "experts.",
            "moe.experts",
            "block_sparse_moe.experts",
            "mlp.experts",
            "feed_forward.experts"
        ]

        # Exclude if it's an expert weight
        for pattern in expert_patterns:
            if pattern in weight_name:
                return False

        # Include if it matches shared patterns
        for pattern in shared_patterns:
            if pattern in weight_name:
                return True

        # Default: include (conservative - better to load shared than miss it)
        return True


    def _log_configuration(self):
        """Log engine configuration."""
        logger.info("=" * 60)
        logger.info("Engine Configuration:")
        logger.info(f"  Model: {self.model_path}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Dtype: {self.dtype}")
        logger.info(f"  Expert cache: {self.config.expert_cache_size} experts")
        logger.info(f"  Predictive prefetch: {self.config.enable_predictive_prefetch}")
        logger.info(f"  Max sequence length: {self.config.max_seq_len}")
        logger.info(f"  Flash attention: {self.config.use_flash_attention}")
        logger.info("=" * 60)

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stream: bool = False
    ) -> str | Iterator[str]:
        """
        Generate text from prompt.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate (default: config.max_tokens)
            temperature: Sampling temperature (default: config.temperature)
            top_p: Nucleus sampling threshold (default: config.top_p)
            top_k: Top-k sampling parameter (default: config.top_k)
            stream: If True, yield tokens as they are generated

        Returns:
            Generated text (or iterator if stream=True)
        """
        # Use config defaults if not specified
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p
        top_k = top_k or self.config.top_k

        logger.info(f"Generating with prompt: '{prompt[:50]}...'")
        logger.info(f"Parameters: max_tokens={max_tokens}, temp={temperature}, "
                   f"top_p={top_p}, stream={stream}")

        start_time = time.monotonic()

        # Tokenize prompt
        prompt_tokens = self._tokenize(prompt)
        logger.info(f"Prompt tokens: {len(prompt_tokens)}")

        # Prefill phase
        prefill_start = time.monotonic()
        initial_state = self._prefill_phase(prompt_tokens)
        prefill_time = time.monotonic() - prefill_start
        logger.info(f"Prefill completed in {prefill_time:.2f}s "
                   f"({len(prompt_tokens) / prefill_time:.1f} tok/s)")

        # Decode phase
        decode_start = time.monotonic()
        if stream:
            return self._decode_phase_streaming(
                initial_state, max_tokens, temperature, top_p, top_k
            )
        else:
            generated_tokens = self._decode_phase(
                initial_state, max_tokens, temperature, top_p, top_k
            )
            decode_time = time.monotonic() - decode_start

            # Decode tokens to text
            generated_text = self._detokenize(generated_tokens)

            # Log statistics
            total_time = time.monotonic() - start_time
            self._log_generation_stats(
                prompt_tokens=len(prompt_tokens),
                generated_tokens=len(generated_tokens),
                prefill_time=prefill_time,
                decode_time=decode_time,
                total_time=total_time,
                initial_state=initial_state
            )

            # Update global stats
            self.total_generated_tokens += len(generated_tokens)
            self.total_generation_time += decode_time
            self.total_requests += 1

            return generated_text

    def _tokenize(self, text: str) -> torch.Tensor:
        """Tokenize input text."""
        token_ids = self.tokenizer.encode(text, return_tensors="pt")
        return token_ids.to(self.device)

    def _detokenize(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def _prefill_phase(self, prompt_tokens: torch.Tensor) -> GenerationState:
        """
        Prefill phase: Process entire prompt in batch through ALL layers.

        Steps:
        1. Embed tokens
        2. Process through all transformer layers (attention + MoE)
        3. Return final hidden state and KV cache

        Args:
            prompt_tokens: Tokenized prompt [batch, seq_len]

        Returns:
            GenerationState with KV cache and hidden states
        """
        batch_size, seq_len = prompt_tokens.shape

        # Step 1: Embed tokens
        hidden_states = self._embed_tokens(prompt_tokens)  # [batch, seq_len, hidden_dim]

        # Step 2: Process through ALL layers: attention (with residual) then MoE.
        # The KV cache is created on the first layer and threaded through.
        kv_cache = None
        for layer_id in range(self.model_info.num_layers):
            hidden_states, kv_cache = self._apply_attention_layer(
                hidden_states, layer_id, kv_cache, mode="prefill"
            )
            hidden_states = self._apply_moe_layer(hidden_states, layer_id, batch_mode=True)

        # Return initial state for decode
        return GenerationState(
            prompt_tokens=prompt_tokens,
            kv_cache=kv_cache,
            hidden_states=hidden_states[:, -1:, :],  # Last token's hidden state
            generated_tokens=[],
            token_times=[],
            cache_hit_count=0,
            cache_miss_count=0
        )

    def _decode_phase(
        self,
        initial_state: GenerationState,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int
    ) -> List[int]:
        """
        Decode phase: Generate tokens one at a time.

        Steps (per token):
        1. Process through ALL transformer layers (attention + MoE)
        2. Generate next token via sampling
        3. Update KV cache

        Args:
            initial_state: State from prefill phase
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling parameter

        Returns:
            List of generated token IDs
        """
        generated_tokens = []
        current_hidden = initial_state.hidden_states
        kv_cache = initial_state.kv_cache

        for step in range(max_tokens):
            step_start = time.monotonic()

            # All layers for this token (attention + MoE, shared with streaming).
            current_hidden, kv_cache = self._forward_one_token(current_hidden, kv_cache)

            # Generate next token
            next_token = self._sample_next_token(
                current_hidden,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )

            generated_tokens.append(next_token.item())

            # Check for EOS token
            if next_token == self.tokenizer.eos_token_id:
                logger.info(f"EOS token generated at step {step}")
                break

            # Embed next token for next iteration
            # next_token has shape [batch], need to add seq_len dimension to make [batch, 1]
            next_token_hidden = self._embed_tokens(next_token.unsqueeze(1))
            current_hidden = next_token_hidden

            # Track timing
            step_time = time.monotonic() - step_start
            initial_state.token_times.append(step_time)

            if (step + 1) % 10 == 0:
                avg_time = sum(initial_state.token_times[-10:]) / 10
                logger.debug(f"Step {step + 1}: {1.0 / avg_time:.1f} tok/s "
                            f"(last 10 tokens)")

        return generated_tokens

    def _decode_phase_streaming(
        self,
        initial_state: GenerationState,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int
    ) -> Iterator[str]:
        """Streaming version of decode phase - yields tokens as generated."""
        current_hidden = initial_state.hidden_states
        kv_cache = initial_state.kv_cache

        for step in range(max_tokens):
            # Same full per-layer body as _decode_phase (single source of truth).
            current_hidden, kv_cache = self._forward_one_token(current_hidden, kv_cache)

            # Sample next token
            next_token = self._sample_next_token(
                current_hidden,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )

            # Yield token as text
            token_text = self.tokenizer.decode([next_token.item()], skip_special_tokens=True)
            yield token_text

            # Check EOS
            if next_token == self.tokenizer.eos_token_id:
                break

            # Prepare next iteration
            next_token_hidden = self._embed_tokens(next_token.unsqueeze(1))
            current_hidden = next_token_hidden

    def _embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed token IDs to hidden states using real embedding layer from model.

        Loads embedding weights from shared_weights and performs embedding lookup.
        """
        batch_size, seq_len = token_ids.shape

        # Find embedding weights in shared weights
        # Different architectures use different keys
        embed_key = None
        for key in self.shared_weights.keys():
            if "embed_tokens.weight" in key or "wte.weight" in key or "tok_embeddings.weight" in key:
                embed_key = key
                break

        if embed_key is None:
            logger.warning("Embedding weights not found in shared weights, using random embeddings")
            return torch.randn(
                batch_size, seq_len, self.model_info.hidden_size,
                device=self.device,
                dtype=self.dtype
            )

        # Get embedding matrix [vocab_size, hidden_size]
        embedding_weights = self.shared_weights[embed_key]

        # Perform embedding lookup
        # token_ids: [batch, seq_len] -> embeddings: [batch, seq_len, hidden_size]
        embeddings = torch.nn.functional.embedding(token_ids, embedding_weights)

        return embeddings

    def _analyze_expert_frequency(self, expert_indices: torch.Tensor) -> List[int]:
        """
        Analyze which experts are used most frequently during prefill.

        Args:
            expert_indices: Expert assignments [batch, seq_len, top_k]

        Returns:
            List of expert IDs sorted by frequency (most frequent first)
        """
        # Flatten and count expert occurrences
        flat_indices = expert_indices.flatten().cpu().tolist()
        expert_counts = Counter(flat_indices)

        # Sort by frequency
        sorted_experts = [expert_id for expert_id, count in expert_counts.most_common()]

        return sorted_experts

    def _preload_experts(self, expert_ids: List[int], layer_id: int = 0):
        """Preload experts into cache for a specific layer."""
        logger.info(f"Preloading {len(expert_ids)} experts for layer {layer_id}...")
        self.expert_processor.preload_experts(
            expert_ids=expert_ids,
            layer_id=layer_id,
            expert_loader=self._create_expert_loader(layer_id)
        )

    def _create_expert_loader(self, layer_id: int) -> Callable[[int], ExpertFFN]:
        """Create expert loader that checks the module cache before storage.

        Three-tier lookup per request:
        1. Module cache  → (layer_id, expert_id) → ExpertFFN (built once per key)
        2. Three-tier cache → expert_cache.get(layer_id, expert_id) -> tensors
        3. Storage        → _load_expert_from_storage (unavoidable I/O)

        Returns a callable that callers pass to ExpertProcessor._get_expert(),
        which wraps it via expert_cache.get_or_load().
        """

        def load_expert(expert_id: int, _layer_id: int = None) -> ExpertFFN:
            """Load expert by ID. _layer_id arg exists for prefetcher compatibility."""
            key = (layer_id, expert_id)

            # Tier 1: module cache hit
            if key in self._expert_module_cache:
                return self._expert_module_cache[key]

            # Tier 2: three-tier cache (GPU → CPU → Storage)
            try:
                weights, tier = self.expert_cache.get(layer_id, expert_id)
                # weights is a dict of tensors already on GPU; build module once
            except ValueError:
                # Tier 3: storage miss → load from disk
                weights = self._load_expert_from_storage(
                    self._ensure_model_downloaded(),
                    layer_id,
                    expert_id,
                )
                tier = "storage"

            has_gate = "w3.weight" in weights
            expert = ExpertFFN(
                hidden_dim=self.model_info.hidden_size,
                expert_dim=self.model_info.intermediate_size,
                activation="silu",
                has_gate=has_gate,
            )
            expert.w1.weight.data = weights["w1.weight"].to(
                device=self.device, dtype=self.dtype
            )
            expert.w2.weight.data = weights["w2.weight"].to(
                device=self.device, dtype=self.dtype
            )
            if has_gate and expert.w3 is not None:
                expert.w3.weight.data = weights["w3.weight"].to(
                    device=self.device, dtype=self.dtype
                )
            expert.eval()

            # Cache the built module for future hits within this decode pass
            self._expert_module_cache[key] = expert
            return expert

        return load_expert

    def _sample_next_token(
        self,
        hidden_states: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int
    ) -> torch.Tensor:
        """
        Sample next token from hidden states using real LM head.

        Args:
            hidden_states: Hidden states [batch, 1, hidden_dim]
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling parameter

        Returns:
            Next token ID [batch]
        """
        # Project to vocabulary using LM head from shared weights
        # Find LM head weights
        lm_head_key = None
        for key in self.shared_weights.keys():
            if "lm_head.weight" in key or "output.weight" in key:
                lm_head_key = key
                break

        if lm_head_key is None:
            # Fallback: use embedding weights (tied embeddings)
            logger.debug("LM head not found, using tied embeddings")
            for key in self.shared_weights.keys():
                if "embed_tokens.weight" in key or "wte.weight" in key:
                    lm_head_key = key
                    break

        if lm_head_key is None:
            raise RuntimeError("Cannot find LM head or embedding weights for token generation")

        # Get LM head weights [vocab_size, hidden_size]
        lm_head_weights = self.shared_weights[lm_head_key]

        # Project: [batch, 1, hidden] @ [hidden, vocab]^T -> [batch, 1, vocab]
        logits = torch.matmul(hidden_states, lm_head_weights.t())

        # Apply temperature
        logits = logits / temperature

        # Apply top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        # Sample from distribution
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs.squeeze(1), num_samples=1)

        return next_token.squeeze(-1)

    def _log_generation_stats(
        self,
        prompt_tokens: int,
        generated_tokens: int,
        prefill_time: float,
        decode_time: float,
        total_time: float,
        initial_state: GenerationState
    ):
        """Log generation statistics."""
        tokens_per_sec = generated_tokens / decode_time if decode_time > 0 else 0

        # Get cache stats
        processor_stats = self.expert_processor.get_stats()
        cache_hit_rate = processor_stats.get('cache_hit_rate', 0.0)

        logger.info("=" * 60)
        logger.info("Generation Statistics:")
        logger.info(f"  Prompt tokens: {prompt_tokens}")
        logger.info(f"  Generated tokens: {generated_tokens}")
        logger.info(f"  Total tokens: {prompt_tokens + generated_tokens}")
        logger.info(f"  Prefill time: {prefill_time:.2f}s ({prompt_tokens / prefill_time:.1f} tok/s)")
        logger.info(f"  Decode time: {decode_time:.2f}s ({tokens_per_sec:.1f} tok/s)")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  Cache hit rate: {cache_hit_rate:.1%}")
        logger.info(f"  Per-token times: min={min(initial_state.token_times):.3f}s, "
                   f"max={max(initial_state.token_times):.3f}s, "
                   f"avg={sum(initial_state.token_times) / len(initial_state.token_times):.3f}s")
        logger.info("=" * 60)

    def get_statistics(self) -> Dict:
        """
        Get comprehensive engine statistics.

        Returns:
            Dictionary with performance metrics including:
            - average_tokens_per_second
            - cache_hit_rate
            - total_tokens_generated
            - memory_usage_gb
            - total_requests
        """
        avg_tokens_per_sec = (
            self.total_generated_tokens / self.total_generation_time
            if self.total_generation_time > 0 else 0
        )

        # Get processor and cache stats
        processor_stats = self.expert_processor.get_stats()
        if self.cache_manager is not None:
            cache_stats = self.cache_manager.get_cache_stats()
        else:
            cache_stats = {}

        # Get three-tier cache statistics
        three_tier_stats = self.expert_cache.get_stats()

        # Estimate memory usage
        if torch.cuda.is_available():
            memory_usage_gb = torch.cuda.memory_allocated() / 1024**3
        else:
            memory_usage_gb = 0.0

        return {
            "average_tokens_per_second": avg_tokens_per_sec,
            "total_tokens_generated": self.total_generated_tokens,
            "total_requests": self.total_requests,
            "memory_usage_gb": memory_usage_gb,
            "processor_stats": processor_stats,
            "cache_stats": cache_stats,
            "three_tier_cache": three_tier_stats,
            "memory_allocation": {
                "gpu_hot_experts": self.memory_allocation.gpu_hot_expert_count,
                "cpu_warm_experts": self.memory_allocation.cpu_warm_expert_count,
                "storage_cold_experts": self.memory_allocation.ssd_cold_expert_count,
                "gpu_utilization_gb": self.memory_allocation.total_gpu_required / 1024**3,
                "cpu_utilization_gb": self.memory_allocation.total_cpu_required / 1024**3,
            },
            "model_info": {
                "model_id": self.model_info.model_id,
                "num_layers": self.model_info.num_layers,
                "num_experts": self.model_info.num_experts,
                "num_experts_per_tok": self.model_info.num_experts_per_tok,
            }
        }

    def _cleanup(self):
        """Release resources after generation (called between requests)."""
        # Clear caches
        if hasattr(self, 'attention_engine') and hasattr(self.attention_engine, 'clear_cache'):
            self.attention_engine.clear_cache()

        # Optionally clear expert cache (or keep hot experts)
        # self.expert_cache.clear()

        # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.debug("Cleanup completed")

    def __del__(self):
        """Cleanup on deletion."""
        self._cleanup()

        # Shutdown cache manager
        if hasattr(self, 'cache_manager'):
            self.cache_manager.shutdown()


__all__ = ["CustomMoEInferenceEngine", "InferenceConfig", "GenerationStats"]
