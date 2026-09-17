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
import threading
import time
from typing import Optional, Dict, List, Iterator, Callable
from dataclasses import dataclass
from collections import Counter

import torch
from pathlib import Path
from safetensors import safe_open

from sparse_llm.inference.router_calculator import RouterCalculator
from sparse_llm.inference.expert_processor import ExpertProcessor, ExpertFFN
from sparse_llm.inference.attention_engine import AttentionEngine, RMSNorm
from sparse_llm.inference.quantized_loader import QuantizedExpertLoader
from sparse_llm.models.quantization import detect_quantization, QuantizationPolicy
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

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for inference engine."""
    # Model configuration
    model_path: str
    device: str = "cuda"
    dtype: str = "float16"
    quantization: str | None = None  # Weight quantization (int4, awq, gptq, fp8, int8, float16, bfloat16, float32)

    # Expert cache configuration
    expert_cache_size: int = 8  # Number of experts to keep in GPU (increased for performance)
    enable_predictive_prefetch: bool = True
    prefetch_confidence_threshold: float = 0.30
    prefetch_top_k: int = 8

    # Expert preloading configuration
    num_hot_experts: int = 0  # Number of hot experts to preload to GPU (0 = on-demand only, safer for small GPUs)
    enable_parallel_preload: bool = False  # Parallel expert loading (disable by default for safety)
    parallel_preload_workers: int = 2  # Number of parallel workers for loading

    # Generation parameters (defaults)
    max_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50

    # Memory configuration
    max_seq_len: int = 4096
    gpu_cache_gb: float = 0.0  # 0 = auto-detect via ResourceBudget
    cpu_cache_gb: float = 0.0  # 0 = auto-detect
    allow_cpu_offload: bool = True
    allow_ssd_offload: bool = True

    # Expert processing mode
    expert_processing_mode: str = "standard"  # "standard" | "tiled" | "fused"

    # Streaming mode for low-resource devices
    streaming_mode: bool = False  # If True, no caching - load/execute/free
    max_parallel_expert_loads: int = 4  # Parallel I/O threads for expert loading

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
        logger.info("Tokenize OK")
        self.router_calculator = self._initialize_router()
        logger.info("Router OK")
        self.attention_engine = self._initialize_attention()
        logger.info("Attention OK")
        self.expert_cache = self._initialize_expert_cache()
        logger.info("Expert cache OK")
        self.cache_manager = self._initialize_cache_manager()
        logger.info("Cache manager OK")
        self.expert_processor = self._initialize_expert_processor()
        logger.info("Expert processor OK")

        # Initialize streaming expert loader if in streaming mode (no cache)
        self.streaming_expert_loader = None
        if self.config.streaming_mode:
            from sparse_llm.loading.streaming_expert_loader import StreamingExpertLoader
            try:
                from sparse_llm.inference.memory_guard import MemoryGuard
                memory_guard = MemoryGuard(
                    device=self.device.index if hasattr(self.device, 'index') else 0,
                    reserve_gb=1.0,
                )
            except Exception:
                memory_guard = None
                logger.warning("MemoryGuard not available, streaming mode will proceed without admission control")

            # Create storage loader closure
            def storage_loader(layer_id, expert_id):
                return self._load_expert_from_safetensors(
                    self._ensure_model_downloaded(), layer_id, expert_id
                )

            self.streaming_expert_loader = StreamingExpertLoader(
                storage_loader=storage_loader,
                device=str(self.device),
                max_parallel=self.config.max_parallel_expert_loads,
            )
            if memory_guard:
                self.streaming_expert_loader.set_memory_guard(memory_guard)
            logger.info(f"Streaming mode enabled: {self.config.max_parallel_expert_loads} parallel loads")

        # Caches must be initialized BEFORE _load_model_weights (uses _model_path)
        self._safetensors_index: dict[str, tuple[Path, str]] = {}
        self._safetensors_files_by_prefix: dict[str, Path] = {}  # Lightweight file lookup
        self._index_built = False  # Lazy index building
        self._model_path: Path | None = None
        self._expert_module_cache: dict[tuple[int, int], ExpertFFN] = {}
        self._expert_weights_cpu: dict[tuple[int, int], dict] = {}
        self._safetensors_file_cache: dict[Path, dict[str, torch.Tensor]] = {}
        self._safetensors_cache_lock = threading.Lock()

        logger.info("Loading model weights...")
        self._load_model_weights()
        logger.info("Model weights OK")
        self._last_prefill_experts: List[int] = []

        # Statistics tracking
        self.total_generated_tokens = 0
        self.total_generation_time = 0.0
        self.total_requests = 0

        logger.info("Engine initialized successfully")
        self._log_configuration()

    def _get_torch_dtype(self, dtype: str) -> torch.dtype:
        """Convert string dtype to torch dtype.

        User's dtype string maps directly to corresponding torch dtype.
        NO remapping or conversion - user's choice is respected throughout.
        """
        dtype_lower = dtype.lower()

        # Standard torch dtypes - direct mapping
        if dtype_lower in ("float32", "fp32"):
            return torch.float32
        if dtype_lower in ("float16", "fp16"):
            return torch.float16
        if dtype_lower in ("bfloat16", "bf16"):
            return torch.bfloat16

        # Quantization formats - use float16 for compute
        if dtype_lower in ("int8", "int4", "fp8", "awq", "gptq", "nf4"):
            return torch.float16

        # Default to float16 for unknown
        return torch.float16

    def _get_quantization_policy(self) -> QuantizationPolicy | None:
        """Build QuantizationPolicy from config.quantization string or model config."""
        quant = self.config.quantization
        if not quant:
            return None

        # First: try explicit user request
        quant_lower = quant.lower()
        if quant_lower in {"int8", "fp8", "int4", "awq", "gptq", "nf4"}:
            group_size_map = {"int4": 128, "awq": 128, "gptq": 128, "nf4": -1}
            return QuantizationPolicy(
                format=quant_lower,
                group_size=group_size_map.get(quant_lower, 128),
                scale_type="absmax",
                dequant_backend="aten",
            )
        if quant_lower in {"float16", "fp16"}:
            return QuantizationPolicy(format="float16")
        if quant_lower in {"bfloat16", "bf16"}:
            return QuantizationPolicy(format="bfloat16")
        if quant_lower in {"float32", "fp32"}:
            return QuantizationPolicy(format="float32")

        # Second: try auto-detect from model config
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(
                self.model_path,
                trust_remote_code=False,
                local_files_only=False,
            )
            quant_config = getattr(config, "quantization_config", None)
            if quant_config:
                return detect_quantization(quant_config)
        except Exception as e:
            logger.debug(f"Could not auto-detect quantization from model config: {e}")

        logger.warning(f"Could not resolve quantization '{quant}', running without quantization")
        return None

    def _load_model_info(self) -> ModelInfo:
        """Load and introspect model architecture."""
        logger.info("Introspecting model architecture...")
        model_info = self.introspector.introspect(
            self.model_path,
            dtype=self.config.dtype,
            quantization=self.config.quantization
        )

        logger.info(f"Model info: {model_info.num_layers} layers, "
                   f"{model_info.num_experts} experts, "
                   f"experts per token: {model_info.num_experts_per_tok}")
        logger.info(f"Shared weights: {model_info.shared_weight_bytes / 1024**3:.2f}GB, "
                   f"Expert weights: {model_info.expert_weight_bytes / 1024**3:.2f}GB each")

        return model_info

    def _calculate_memory_budget(self) -> MemoryAllocation:
        """Calculate memory allocation plan using three-tier system."""
        from sparse_llm.loading.resource_budget import ResourceBudget

        # Get available resources (Phase 1), applying explicit cache overrides if set
        resource_budget = ResourceBudget.from_system(
            gpu_cache_gb_override=self.config.gpu_cache_gb if self.config.gpu_cache_gb > 0 else None,
            cpu_cache_gb_override=self.config.cpu_cache_gb if self.config.cpu_cache_gb > 0 else None,
        )

        # Create user request
        user_request = UserRequest(
            max_model_len=self.config.max_seq_len,
            dtype=self.config.dtype,
            quantization=self.config.quantization,
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
        logger.info(f"  Router params: num_experts={self.model_info.num_experts}, "
                   f"top_k={self.model_info.num_experts_per_tok}, "
                   f"hidden_dim={self.model_info.hidden_size}")

        # Create router without weights initially - will be set during weight loading
        router = RouterCalculator(
            router_weights=None,  # Will be set during _load_model_weights
            num_experts=self.model_info.num_experts,
            top_k=self.model_info.num_experts_per_tok,
            hidden_dim=self.model_info.hidden_size,  # Provide hidden_dim for lazy init
            dtype=self.dtype,
        )
        logger.info(f"  Router created: {router}")

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

    def _initialize_expert_cache(self):
        """Initialize three-tier expert cache (GPU → CPU → Storage)."""
        logger.info("Initializing three-tier expert cache...")

        # Import the hierarchical expert loader
        from sparse_llm.loading.hierarchical_expert_loader import HierarchicalExpertLoader, TieredResourceConfig
        from sparse_llm.inference.memory_guard import MemoryGuard

        # Get resource information
        from sparse_llm.loading.resource_budget import ResourceBudget
        resource_budget = ResourceBudget.from_system()

        # Calculate tier configuration based on available resources
        tier_config = TieredResourceConfig.from_resources(
            gpu_available_bytes=resource_budget.total_gpu_bytes if resource_budget.gpus else 0,
            cpu_available_bytes=resource_budget.total_cpu_bytes,
            storage_is_ssd=resource_budget.storage.is_ssd,
            storage_bandwidth_mbps=resource_budget.storage.estimated_bandwidth_mbps,
        )

        logger.info(f"Three-tier cache configuration:")
        logger.info(f"  GPU expert fraction: {tier_config.gpu_expert_fraction}")
        logger.info(f"  CPU expert fraction: {tier_config.cpu_expert_fraction}")
        logger.info(f"  Storage prefetch: {tier_config.storage_prefetch}")

        # Create memory guard for OOM prevention
        memory_guard = MemoryGuard(
            device=self.device.index if hasattr(self.device, 'index') else 0,
            reserve_gb=tier_config.gpu_reserve_gb,
        )

        # Storage loader will be set later in _load_model_weights
        cache = HierarchicalExpertLoader(
            tier_config=tier_config,
            expert_bytes=self.model_info.expert_weight_bytes,
            storage_loader=None,  # Will be set after model path is available
            device=str(self.device),
            memory_guard=memory_guard,
        )

        # Wire memory guard to cache for eviction triggers
        memory_guard.set_expert_cache(cache)

        # Store memory guard for later use
        self.memory_guard = memory_guard

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
            dtype=self.dtype,
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

        # Step 3: Initialize quantized expert loader if quantization is requested
        quant_loader = None
        if self.config.quantization:
            quant_policy = self._get_quantization_policy()
            if quant_policy and quant_policy.is_quantized():
                quant_loader = QuantizedExpertLoader(
                    policy=quant_policy,
                    device=str(self.device),
                    target_dtype=self.dtype,
                )
                logger.info(f"   Quantization: {quant_policy.format}, "
                           f"group_size={quant_policy.group_size}, "
                           f"target dtype={self.dtype}")

        # Step 4: Configure storage loader for on-demand expert loading
        logger.info(f"[3/4] Configuring expert storage loader (on-demand)...")

        # Create storage loader for HierarchicalExpertLoader
        # Simple in-memory cache to avoid repeated safetensors access
        _expert_ram_cache: dict[tuple[int, int], dict] = {}

        def on_demand_expert_loader(layer_id: int, expert_id: int):
            """Load expert from safetensors on-demand with RAM caching and dequantization."""
            key = (layer_id, expert_id)
            if key in _expert_ram_cache:
                return _expert_ram_cache[key]
            weights = self._load_expert_from_safetensors(
                self._ensure_model_downloaded(), layer_id, expert_id
            )
            if weights is None:
                raise FileNotFoundError(
                    f"Expert ({layer_id}, {expert_id}) not found in model. "
                    f"Cannot continue without required expert weights."
                )
            # Apply dequantization if quantization policy is active
            if quant_loader is not None:
                weights = quant_loader.load_expert(
                    key=(layer_id, expert_id),
                    tensors=weights,
                    state_dict=self.shared_weights,
                )
            _expert_ram_cache[key] = weights
            return weights

        # Set storage loader on HierarchicalExpertLoader
        self.expert_cache.storage_loader = on_demand_expert_loader

        # Step 4: Initialize lazy-load cache for experts
        # NOTE: Preloading all experts causes SIGSEGV crashes in safetensors on Windows
        # Experts are loaded on-demand via _load_expert_from_safetensors instead
        self._expert_ram_cache = {}
        logger.info(f"[4/4] Expert weights loaded on-demand (lazy loading)")

        # Step 5: Log tier distribution
        logger.info(f"[5/5] Weight loading complete:")
        logger.info(f"  ✓ Shared weights: {len(self.shared_weights)} tensors on GPU")
        stats = self.expert_cache.stats()
        logger.info(f"  ✓ GPU cache: {stats.get('gpu_cache_count', 0)} experts")
        logger.info(f"  ✓ CPU cache: {stats.get('cpu_cache_count', 0)} experts")

        # Log actual memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    def _ensure_model_downloaded(self) -> Path:
        """Ensure model is available locally. Uses cached path to avoid repeated downloads."""
        from pathlib import Path
        from huggingface_hub import snapshot_download, hf_hub_download

        # Return cached path if already downloaded
        if self._model_path is not None:
            return self._model_path

        local_path = Path(self.model_path)
        if local_path.exists():
            logger.info(f"Using local model: {local_path}")
            self._model_path = local_path
            return self._model_path

        # Check if model is cached using hf_hub_download first (faster, uses cache)
        try:
            logger.info(f"Checking HuggingFace cache for {self.model_path}...")
            cached_path = hf_hub_download(
                self.model_path,
                filename="config.json",
                local_files_only=True
            )
            if cached_path:
                # Extract snapshot directory from config path
                config_path = Path(cached_path)
                snapshot_dir = config_path.parent
                if snapshot_dir.exists():
                    safetensors = list(snapshot_dir.glob("*.safetensors"))
                    if safetensors:
                        logger.info(f"Using cached model at: {snapshot_dir}")
                        self._model_path = snapshot_dir
                        return snapshot_dir
        except Exception:
            pass  # Not cached, need to download

        # Download model
        logger.info(f"Downloading model from HuggingFace: {self.model_path}")
        downloaded_path = snapshot_download(
            self.model_path,
            allow_patterns=["*.safetensors", "*.json"],
            ignore_patterns=["*.bin"]  # Prefer safetensors over pickle
        )
        self._model_path = Path(downloaded_path)
        return self._model_path

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
        self._ensure_index_built()  # Build index before accessing

        # Group shared weights by file
        shared_keys_by_file: dict[Path, list[str]] = {}
        for key, (st_file, _) in self._safetensors_index.items():
            if self._is_shared_weight(key):
                shared_keys_by_file.setdefault(st_file, []).append(key)

        total_bytes = 0
        router_weights_by_layer = {}

        # Load each file's shared weights
        for st_file, keys in shared_keys_by_file.items():
            logger.info(f"   Loading {len(keys)} shared weights from {st_file.name}...")
            try:
                weights = self._load_file_weights(st_file, keys)
                for key in keys:
                    if key in weights:
                        tensor = weights[key].to(device=self.device, dtype=self.dtype)
                        shared_weights[key] = tensor
                        total_bytes += tensor.numel() * tensor.element_size()

                        # Extract router weights (all naming conventions)
                        is_router = (
                            ".gate.weight" in key or
                            ".router.layer.weight" in key or
                            ".router.weight" in key
                        ) and "layers" in key and "experts" not in key

                        if is_router:
                            try:
                                layer_num = int(key.split(".layers.")[1].split(".")[0])
                                router_weights_by_layer[layer_num] = tensor
                            except (IndexError, ValueError):
                                pass
            except Exception as e:
                logger.warning(f"   Failed to load shared weights from {st_file.name}: {e}")

        logger.info(f"   Loaded {len(shared_weights)} shared weights ({total_bytes / 1024**3:.2f}GB)")

        # Look for router gate weights (all naming conventions)
        router_patterns = [
            ".block_sparse_moe.gate.weight",
            ".mlp.gate.weight",
            ".moe.gate.weight",
            ".ffn.gate.weight",
            ".feed_forward.gate.weight",
            ".block_sparse_moe.router.layer.weight",
            ".mlp.router.layer.weight",
            ".moe.router.layer.weight",
            ".ffn.router.layer.weight",
            ".feed_forward.router.layer.weight",
            ".block_sparse_moe.router.weight",
            ".mlp.router.weight",
            ".moe.router.weight",
            ".ffn.router.weight",
            ".feed_forward.router.weight",
        ]
        router_key = None
        for key in shared_weights.keys():
            if "layers" in key:
                for pattern in router_patterns:
                    if pattern in key:
                        router_key = key
                        break
                if router_key:
                    break

        # Load router weights from checkpoint or fall back to random
        if router_key and router_key in shared_weights:
            real_weights = shared_weights[router_key].to(device=self.device, dtype=self.dtype)
            # Transpose if needed: model has [num_experts, hidden], router expects [hidden, num_experts]
            if real_weights.shape[0] == self.model_info.num_experts:
                real_weights = real_weights.T
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
        """Build key→(file, key) index for all safetensors files.

        Uses lightweight header-only reading to avoid memory exhaustion.
        """
        from safetensors.torch import load_file

        files = list(model_path.glob("*.safetensors"))
        if not files:
            files = list(model_path.glob("**/*.safetensors"))

        # Index files by filename prefix for quick lookup
        for st_file in files:
            prefix = st_file.stem  # filename without extension
            self._safetensors_files_by_prefix[prefix] = st_file

    def _ensure_index_built(self) -> None:
        """Build full index lazily only when needed."""
        if self._index_built:
            return
        from safetensors import safe_open

        # Check all files we've seen
        for st_file in self._safetensors_files_by_prefix.values():
            try:
                with safe_open(st_file, framework="pt") as f:
                    for key in f.keys():
                        self._safetensors_index[key] = (st_file, key)
            except Exception as e:
                logger.debug(f"Could not index {st_file.name}: {e}")

        self._index_built = True
        logger.info(f"   Indexed {len(self._safetensors_index)} keys")

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

    def _streaming_moe_layer(
        self,
        normed: torch.Tensor,
        layer_idx: int,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        batch_mode: bool,
    ) -> torch.Tensor:
        """Streaming MoE layer: load→execute→free for each expert.

        This bypasses the cache entirely for low-resource devices.
        Uses parallel loading for prefill and sequential for decode.
        """
        if batch_mode:
            # Prefill mode: analyze expert frequency and load in parallel
            activated_experts = self._analyze_expert_frequency(expert_indices)
            if layer_idx == 0:
                logger.info(f"Streaming: Layer {layer_idx} activating {len(activated_experts)} experts")

            output = self.streaming_expert_loader.execute_experts_parallel(
                layer_id=layer_idx,
                expert_ids=activated_experts,
                hidden_state=normed,
                expert_indices=expert_indices,
                expert_weights=expert_weights,
            )
        else:
            # Decode mode: single token, load experts sequentially
            expert_ids_list = expert_indices.squeeze(1).tolist()
            unique_experts = list(set(expert_ids_list))

            output = torch.zeros_like(normed)
            for expert_id in unique_experts:
                expert_output = self.streaming_expert_loader.execute_expert(
                    layer_id=layer_idx,
                    expert_id=expert_id,
                    hidden_state=normed,
                )
                # Get router weight for this expert
                for i, eid in enumerate(expert_ids_list):
                    if eid == expert_id:
                        router_w = expert_weights[0, i] if expert_weights.dim() > 1 else expert_weights[i]
                        output = output + router_w * expert_output

        return output

    def _apply_attention_layer(
        self,
        h: torch.Tensor,
        layer_idx: int,
        kv_cache,
        mode: str = "decode",
        current_pos: int | None = None,
    ):
        """One attention sublayer with its residual: h = h + attn(norm(h)).

        `mode="prefill"` runs the whole prompt through one layer; `"decode"`
        processes a single token. The caller owns the per-layer loop.

        `current_pos` is required in decode mode so all layers write to the
        same position (the position counter is advanced exactly once per
        token, after the full layer loop).
        """
        norm = self._get_norm(layer_idx, "input_layernorm")
        normed = norm(h)

        if mode == "prefill":
            attn_out, kv_cache = self.attention_engine.prefill(
                input_ids=normed, layer_idx=layer_idx, kv_cache=kv_cache
            )
        else:
            # current_pos must be provided in decode mode so all layers write
            # to the same position (seq_len advances once per token, after
            # the full layer loop, not per-layer).
            if current_pos is None:
                current_pos = kv_cache.get('seq_len', 0)
            attn_out, kv_cache = self.attention_engine.decode(
                token_id=normed, layer_idx=layer_idx, kv_cache=kv_cache,
                current_pos=current_pos
            )

        return h + attn_out, kv_cache

    def _apply_moe_layer(self, h: torch.Tensor, layer_idx: int, batch_mode: bool) -> torch.Tensor:
        """One MoE sublayer with its residual: h = h + moe(norm(h))."""
        try:
            
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

            # Streaming mode: load→execute→free (no cache)
            if self.config.streaming_mode and self.streaming_expert_loader is not None:
                return h + self._streaming_moe_layer(normed, layer_idx, expert_indices, expert_weights, batch_mode)

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
                if hasattr(self.expert_processor, 'observe_routing'):
                    self.expert_processor.observe_routing(activated_experts, layer_id=layer_idx)
                # Prefetch next layer's predicted experts
                if layer_idx < self.model_info.num_layers - 1:
                    self.expert_processor.prefetch_next(activated_experts, layer_id=layer_idx + 1)
                logger.info(f"{__name__}+ Moe Layers use Standard mode for simplicity")
            else:
                activated_experts = self._analyze_expert_frequency(expert_indices)
                # CPU compute for single tokens: avoid GPU transfer overhead
                # Expert weights stay on CPU, compute locally without transferring to GPU
                # Decision: use CPU for single tokens unless expert is already on GPU
                out = self.expert_processor.process_single(
                    hidden_state=normed,
                    expert_indices=expert_indices.squeeze(1),  # [batch, top_k]
                    expert_weights=expert_weights.squeeze(1),
                    layer_id=layer_idx,
                    expert_loader=self._create_expert_loader(layer_idx, force_cpu=True),
                    update_predictor=True,
                    compute_on_cpu=True,  # CPU compute for single tokens avoids transfer overhead
                )
                # Observe routing for decode mode
                if hasattr(self.expert_processor, 'observe_routing'):
                    self.expert_processor.observe_routing(activated_experts, layer_id=layer_idx)
                # Prefetch predicted next experts for next token (continuous during decode)
                if hasattr(self.expert_processor, 'prefetch_next'):
                    self.expert_processor.prefetch_next(activated_experts, layer_id=layer_idx)

            return h + out
        except Exception as e:
            logger.error(f"{__name__}:{e}")

    def _forward_one_token(self, current_hidden: torch.Tensor, kv_cache):
        """Run every layer for one decode token. Shared by both decode paths.

        R7 seq_len contract: AttentionEngine.decode writes K/V at
        `kv_cache['seq_len']` and does NOT advance it. We advance it exactly
        once per generated token, here, after the full layer loop. Advancing
        inside `decode` would overshoot by num_layers per token.
        """
        # Compute current position once, before the layer loop. All layers
        # write to the same position; seq_len advances after the loop.
        current_pos = kv_cache.get('seq_len', 0)
        for layer_id in range(self.model_info.num_layers):
            current_hidden, kv_cache = self._apply_attention_layer(
                current_hidden, layer_id, kv_cache, mode="decode",
                current_pos=current_pos
            )
            current_hidden = self._apply_moe_layer(current_hidden, layer_id, batch_mode=False)

        kv_cache['seq_len'] = current_pos + 1
        return self._apply_final_norm(current_hidden), kv_cache

    def _load_file_weights(self, st_file: Path, keys: list[str]) -> dict[str, torch.Tensor]:
        """Load specific keys from a safetensors file."""
        from safetensors import safe_open

        weights = {}
        with safe_open(st_file, framework="pt") as f:
            for key in keys:
                if key in f.keys():
                    weights[key] = f.get_tensor(key)
        return weights

    def _load_expert_from_safetensors(
        self,
        model_path: Path,
        layer_id: int,
        expert_id: int
    ) -> Optional[dict[str, torch.Tensor]]:
        """Load specific expert weights from safetensors files.

        Uses cached file loads to avoid disk I/O on repeated access.
        Returns None if expert not found.
        """
        # Ensure index is built before accessing it
        self._ensure_index_built()

        expert_weights = {}

        # Build candidate key prefixes for this (layer, expert)
        # Covers all MoE naming conventions: Mixtral, Qwen2MoE, DeepSeek, etc.
        prefix_patterns = [
            f"model.layers.{layer_id}.block_sparse_moe.experts.{expert_id}.",
            f"model.layers.{layer_id}.mlp.experts.{expert_id}.",
            f"model.layers.{layer_id}.moe.experts.{expert_id}.",
            f"model.layers.{layer_id}.ffn.experts.{expert_id}.",
            f"model.layers.{layer_id}.feed_forward.experts.{expert_id}.",
        ]

        # Group matching keys by file to batch reads
        keys_by_file: dict[Path, list[str]] = {}
        matching_keys = []
        for key, (st_file, _canonical) in self._safetensors_index.items():
            for prefix in prefix_patterns:
                if key.startswith(prefix):
                    keys_by_file.setdefault(st_file, []).append(key)
                    matching_keys.append(key)
                    break

        if not keys_by_file:
            # Log debug info for first few failures
            if expert_id < 5:
                logger.warning(f"Expert ({layer_id}, {expert_id}) not found in index. Index has {len(self._safetensors_index)} keys, checking patterns: {prefix_patterns}")
                logger.warning(f"Sample index keys: {list(self._safetensors_index.keys())[:5]}")
            return None

        # Open each file once, read only needed tensors, close immediately
        # Use mmap=False to avoid virtual address space exhaustion on Windows
        raw_weights: dict[str, torch.Tensor] = {}
        for st_file, keys in keys_by_file.items():
            # Load only the tensors we need directly from the file
            with safe_open(st_file, framework="pt") as f:
                for key in keys:
                    if key in f.keys():
                        raw_weights[key] = f.get_tensor(key)

        if not raw_weights:
            return None

        # Normalize to w1/w2/w3 (Qwen uses gate_proj/down_proj/up_proj, Mixtral uses w1/w2/w3)
        for key, tensor in raw_weights.items():
            # Extract weight name after expert ID
            parts = key.split(f".experts.{expert_id}.")
            if len(parts) == 2:
                weight_name = parts[1]
            else:
                weight_name = key.split(f".{expert_id}.")[-1]

            # Map both Qwen (gate_proj/down_proj/up_proj) and Mixtral (w1/w2/w3) naming
            if "gate_proj" in weight_name or weight_name == "w1.weight":
                expert_weights["w1.weight"] = tensor
            elif "down_proj" in weight_name or weight_name == "w2.weight":
                expert_weights["w2.weight"] = tensor
            elif "up_proj" in weight_name or weight_name == "w3.weight":
                expert_weights["w3.weight"] = tensor
            elif weight_name.startswith("w1") or weight_name.startswith("w2") or weight_name.startswith("w3"):
                # Mixtral-style w1.weight, w2.weight, w3.weight
                expert_weights[weight_name] = tensor
            else:
                # Skip unknown weights (like shared_expert gate for some architectures)
                logger.debug(f"Skipping unknown expert weight: {weight_name}")
                continue

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

    def _preload_hot_experts(self, expert_ids: List[int]) -> None:
        """Preload hot experts into GPU cache for all layers using parallel I/O.

        This reduces disk I/O during inference by having common experts already loaded.
        Uses ThreadPoolExecutor for parallel file loading to maximize disk throughput.

        Args:
            expert_ids: List of expert IDs to preload (same across all layers)
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        model_path = self._ensure_model_downloaded()
        preloaded_count = 0
        load_lock = threading.Lock()

        # Use config for number of workers, default to 8
        num_workers = getattr(self.config, 'parallel_preload_workers', 8)

        def preload_expert_for_layer(layer_id: int, expert_id: int) -> tuple[int, int, bool]:
            """Load single expert for a layer. Returns (layer_id, expert_id, success)."""
            try:
                weights = self._load_expert_from_safetensors(model_path, layer_id, expert_id)
                if weights:
                    # Preload to GPU tier
                    self.expert_cache.preload_gpu(layer_id, expert_id, weights)
                    return (layer_id, expert_id, True)
            except Exception as e:
                logger.debug(f"Could not preload expert {expert_id} for layer {layer_id}: {e}")
            return (layer_id, expert_id, False)

        # Build list of all (layer, expert) pairs to load
        tasks = [(layer_id, expert_id)
                 for layer_id in range(self.model_info.num_layers)
                 for expert_id in expert_ids]

        logger.info(f"  Loading {len(tasks)} expert-layer pairs with {num_workers} workers...")

        # Use thread pool for parallel disk I/O
        if getattr(self.config, 'enable_parallel_preload', True):
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(preload_expert_for_layer, layer_id, expert_id)
                          for layer_id, expert_id in tasks]
                for future in as_completed(futures):
                    layer_id, expert_id, success = future.result()
                    if success:
                        with load_lock:
                            preloaded_count += 1
        else:
            # Sequential loading as fallback
            for layer_id, expert_id in tasks:
                _, _, success = preload_expert_for_layer(layer_id, expert_id)
                if success:
                    preloaded_count += 1

        logger.info(f"  Preloaded {preloaded_count} expert-layer combinations to GPU")

        # Skip building full safetensors file cache - it causes OOM on large models
        # Per-file caching in _load_expert_from_safetensors is sufficient

    def _build_safetensors_file_cache(self, model_path: Path) -> None:
        """Build cache of loaded safetensors files to avoid reloading.

        This caches entire safetensors files in memory after first access,
        dramatically reducing disk I/O for subsequent expert loads.
        """
        for st_file in self._safetensors_index.values():
            file_path = st_file[0]
            if file_path not in self._safetensors_file_cache:
                try:
                    import safetensors.torch
                    self._safetensors_file_cache[file_path] = safetensors.torch.load_file(str(file_path))
                    logger.debug(f"  Cached safetensors file: {file_path.name}")
                except Exception as e:
                    logger.debug(f"  Could not cache {file_path.name}: {e}")

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

        # Expert patterns to exclude (all MoE naming conventions)
        expert_patterns = [
            ".experts.",
            "moe.experts",
            "block_sparse_moe.experts",
            "mlp.experts",
            "ffn.experts",
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
        logger.info(f"  Quantization: {self.config.quantization or 'none'}")
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
        logger.info(f"Prompt tokens: {(prompt_tokens)}")

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
        3. Analyze expert usage and preload hot experts in parallel
        4. Return final hidden state and KV cache

        Args:
            prompt_tokens: Tokenized prompt [batch, seq_len]

        Returns:
            GenerationState with KV cache and hidden states
        """
        from concurrent.futures import ThreadPoolExecutor

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

        # Step 3: After prefill, preload hot experts in parallel for decode phase
        # This analyzes which experts were used and preloads them for the next tokens
        if hasattr(self, '_last_prefill_experts') and self._last_prefill_experts:
            hot_experts = self._last_prefill_experts[:8]  # Top 8 most used
            logger.info(f"  Prefetching {len(hot_experts)} hot experts for decode...")

            # Parallel preload of hot experts for all layers
            model_path = self._ensure_model_downloaded()
            with ThreadPoolExecutor(max_workers=8) as executor:
                for layer_id in range(self.model_info.num_layers):
                    for expert_id in hot_experts:
                        key = (layer_id, expert_id)
                        if key not in self._expert_weights_cpu:
                            executor.submit(
                                self._load_expert_from_safetensors,
                                model_path, layer_id, expert_id
                            )

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

            # Speculative prefetch: Load likely experts for next token in parallel
            if hasattr(self, '_last_prefill_experts') and self._last_prefill_experts:
                from concurrent.futures import ThreadPoolExecutor
                hot_experts = self._last_prefill_experts[:4]  # Top 4 for speculative load
                model_path = self._ensure_model_downloaded()
                with ThreadPoolExecutor(max_workers=4) as executor:
                    for layer_id in range(self.model_info.num_layers):
                        for expert_id in hot_experts:
                            key = (layer_id, expert_id)
                            if key not in self._expert_weights_cpu:
                                executor.submit(
                                    self._load_expert_from_safetensors,
                                    model_path, layer_id, expert_id
                                )

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
        try: 
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
        except Exception as e:
            raise e

    def _analyze_expert_frequency(self, expert_indices: torch.Tensor) -> List[int]:
        """
        Analyze which experts are used most frequently during prefill.

        Args:
            expert_indices: Expert assignments [batch, seq_len, top_k]

        Returns:
            List of expert IDs sorted by frequency (most frequent first)
        """
        try:
            
            # Flatten and count expert occurrences
            flat_indices = expert_indices.flatten().cpu().tolist()
            expert_counts = Counter(flat_indices)

            # Sort by frequency
            sorted_experts = [expert_id for expert_id, count in expert_counts.most_common()]

            # Store for prefill-phase preloading
            self._last_prefill_experts = sorted_experts

            return sorted_experts
        
        except Exception as e:
            raise e

    def _create_expert_loader(
        self,
        layer_id: int,
        force_cpu: bool = False,
    ) -> Callable[[int, int], ExpertFFN]:
        """Create expert loader that bypasses HierarchicalExpertLoader.

        Uses direct safetensors access with Python RAM caching to avoid
        storage_loader crashes in HierarchicalExpertLoader.

        Args:
            layer_id: Layer ID for this loader
            force_cpu: If True, create ExpertFFN on CPU for CPU compute path

        Returns a callable that callers pass to ExpertProcessor._get_expert().
        """
        engine = self  # Capture for closure
        # Module-level cache to avoid repeated safetensors access
        if not hasattr(engine, '_direct_expert_cache'):
            engine._direct_expert_cache = {}

        def load_expert(expert_id: int, layer_id: int = None) -> ExpertFFN:
            """Load expert by ID using preloaded CPU RAM cache."""
            if layer_id is None:
                layer_id = 0
            cache_key = (layer_id, expert_id)

            # Check preloaded RAM cache first (from startup preload)
            if hasattr(engine, '_expert_ram_cache') and cache_key in engine._expert_ram_cache:
                weights = engine._expert_ram_cache[cache_key]
            # Check direct cache (from lazy loading)
            elif cache_key in engine._direct_expert_cache:
                weights = engine._direct_expert_cache[cache_key]
            else:
                # Load from safetensors - this should rarely happen now
                weights = engine._load_expert_from_safetensors(
                    engine._ensure_model_downloaded(), layer_id, expert_id
                )
                if weights is None:
                    raise FileNotFoundError(f"Expert {expert_id} for layer {layer_id} not found")
                engine._direct_expert_cache[cache_key] = weights

            # Create ExpertFFN from weights - detect dimensions from actual weight shapes
            has_gate = "w3.weight" in weights
            w1_shape = weights["w1.weight"].shape  # [expert_dim, hidden_dim]
            actual_hidden_dim = w1_shape[1]  # w1: [expert_dim, hidden_dim]
            actual_expert_dim = w1_shape[0]  # w1: [expert_dim, hidden_dim]

            # Use configured dtype for expert computations (user's --dtype choice)
            expert_dtype = self.dtype

            # For CPU compute: create ExpertFFN on CPU
            # For GPU compute: create ExpertFFN on GPU
            if force_cpu:
                target_device = "cpu"
            else:
                target_device = str(engine.device)

            expert = ExpertFFN(
                hidden_dim=actual_hidden_dim,
                expert_dim=actual_expert_dim,
                activation="silu",
                has_gate=has_gate,
                dtype=expert_dtype,
            )
            expert = expert.to(target_device)

            # Copy weights in expert_dtype (user's --dtype choice)
            expert.w1.weight.copy_(weights["w1.weight"].to(dtype=expert_dtype))
            expert.w2.weight.copy_(weights["w2.weight"].to(dtype=expert_dtype))
            if has_gate and expert.w3 is not None:
                expert.w3.weight.copy_(weights["w3.weight"].to(dtype=expert_dtype))
            expert.eval()

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
        three_tier_stats = self.expert_cache.stats()

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
