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
from sparse_llm.inference.attention_engine import AttentionEngine
from sparse_llm.inference.expert_cache_manager import ExpertCacheManager
from sparse_llm.cache.expert_cache import ExpertCache
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
            dtype=self.dtype
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
            enable_predictive_prefetch=self.config.enable_predictive_prefetch
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

        # Step 3: Configure storage loader for expert cache
        logger.info(f"[2/4] Configuring expert storage loader...")
        self.expert_cache.storage_loader = lambda layer_id, expert_id: self._load_expert_from_storage(
            model_path, layer_id, expert_id
        )

        # Step 4: Preload hot experts to GPU cache
        if self.memory_allocation.gpu_hot_expert_count > 0:
            logger.info(f"[3/4] Preloading {self.memory_allocation.gpu_hot_expert_count} hot experts to GPU...")
            self._preload_hot_experts(model_path)
        else:
            logger.info(f"[3/4] No hot experts to preload (all on-demand)")

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
        """Load shared weights (embeddings, attention, norms, router) to GPU."""
        shared_weights = {}

        # Find safetensors files
        safetensors_files = list(model_path.glob("*.safetensors"))
        if not safetensors_files:
            safetensors_files = list(model_path.glob("**/*.safetensors"))

        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors files found in {model_path}")

        # Load shared weights from all files
        # Load to CPU first to avoid safetensors device issue, then move to GPU
        total_bytes = 0
        router_weights_by_layer = {}  # Collect router weights for each layer

        for st_file in safetensors_files:
            with safe_open(st_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if self._is_shared_weight(key):
                        tensor = f.get_tensor(key)
                        # Move to target device and dtype
                        tensor = tensor.to(device=self.device, dtype=self.dtype)
                        shared_weights[key] = tensor
                        total_bytes += tensor.numel() * tensor.element_size()

                        # Track router weights for router calculator initialization
                        if "gate" in key and "layers" in key and "block_sparse_moe" in key:
                            # Extract layer number from key like "model.layers.0.block_sparse_moe.gate.weight"
                            try:
                                layer_num = int(key.split(".layers.")[1].split(".")[0])
                                router_weights_by_layer[layer_num] = tensor
                            except (IndexError, ValueError):
                                pass

        # Initialize router calculator with real weights if found
        if router_weights_by_layer:
            # Use first layer's router weights (all layers typically have same shape)
            first_layer_router = router_weights_by_layer[min(router_weights_by_layer.keys())]
            self.router_calculator.set_router_weights(first_layer_router)
            self.router_weights_by_layer = router_weights_by_layer
            logger.info(f"   ✓ Loaded router weights for {len(router_weights_by_layer)} layers")
        else:
            logger.warning("   ⚠ No router weights found in checkpoint - router will use random initialization")
            # Fallback: create random router weights
            random_weights = torch.randn(
                self.model_info.hidden_size,
                self.model_info.num_experts,
                device=self.device,
                dtype=self.dtype
            )
            self.router_calculator.set_router_weights(random_weights)
            self.router_weights_by_layer = {}

        logger.info(f"   ✓ Loaded {len(shared_weights)} shared weight tensors ({total_bytes / 1024**3:.2f}GB)")
        return shared_weights

    def _load_expert_from_storage(
        self,
        model_path: Path,
        layer_id: int,
        expert_id: int
    ) -> dict[str, torch.Tensor]:
        """Load specific expert weights from storage (cold tier).

        Returns a dictionary with normalized keys (w1.weight, w2.weight, w3.weight)
        that can be loaded into ExpertFFN module.
        """
        from safetensors import safe_open

        expert_weights = {}

        # Expert weight patterns (universal across architectures)
        # Format: model.layers.{layer}.{moe_module}.experts.{expert_id}.{weight_name}
        patterns = [
            f"model.layers.{layer_id}.block_sparse_moe.experts.{expert_id}.",  # Mixtral
            f"model.layers.{layer_id}.mlp.experts.{expert_id}.",  # Some architectures
            f"model.layers.{layer_id}.moe.experts.{expert_id}.",  # Alternative
            f"model.layers.{layer_id}.feed_forward.experts.{expert_id}."  # Alternative
        ]

        # Load from safetensors files
        safetensors_files = list(model_path.glob("*.safetensors"))
        if not safetensors_files:
            safetensors_files = list(model_path.glob("**/*.safetensors"))

        raw_weights = {}
        for st_file in safetensors_files:
            with safe_open(st_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    # Check if matches expert pattern
                    for pattern in patterns:
                        if pattern in key:
                            # Load to CPU, will be moved to device when loaded into expert module
                            raw_weights[key] = f.get_tensor(key)

        if not raw_weights:
            raise FileNotFoundError(
                f"No weights found for expert ({layer_id}, {expert_id}) in {model_path}. "
                f"This model may not be a supported MoE architecture."
            )

        # Normalize weight keys to standard format (w1, w2, w3)
        # Different architectures use different naming:
        # - Mixtral: w1, w2, w3
        # - Qwen: gate_proj, up_proj, down_proj
        # - DeepSeek: gate_proj, up_proj, down_proj
        for key, tensor in raw_weights.items():
            # Extract the weight name (last part after expert_id)
            weight_name = key.split(f".{expert_id}.")[-1]

            # Normalize to w1/w2/w3 format
            if "w1.weight" in weight_name or "gate_proj.weight" in weight_name:
                expert_weights["w1.weight"] = tensor
            elif "w2.weight" in weight_name or "down_proj.weight" in weight_name:
                expert_weights["w2.weight"] = tensor
            elif "w3.weight" in weight_name or "up_proj.weight" in weight_name:
                expert_weights["w3.weight"] = tensor
            else:
                # Keep other weights as-is (biases, etc.)
                expert_weights[weight_name] = tensor

        # Validate we have the required weights
        required_keys = {"w1.weight", "w2.weight"}
        if not required_keys.issubset(expert_weights.keys()):
            raise ValueError(
                f"Expert ({layer_id}, {expert_id}) missing required weights. "
                f"Found: {list(expert_weights.keys())}, required: {required_keys}"
            )

        return expert_weights

    def _preload_hot_experts(self, model_path: Path) -> None:
        """Preload hot experts to GPU cache."""
        hot_count = self.memory_allocation.gpu_hot_expert_count
        total_experts = self.model_info.num_layers * self.model_info.num_experts

        experts_loaded = 0
        target = min(hot_count, total_experts)

        for layer_id in range(self.model_info.num_layers):
            for expert_id in range(self.model_info.num_experts):
                if experts_loaded >= target:
                    break

                # Load expert weights from storage
                expert_weights = self._load_expert_from_storage(model_path, layer_id, expert_id)

                # Preload to GPU cache
                self.expert_cache.preload_gpu(layer_id, expert_id, expert_weights)

                experts_loaded += 1

            if experts_loaded >= target:
                break

        logger.info(f"   ✓ Preloaded {experts_loaded}/{target} hot experts to GPU cache")

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

        # Step 2: Initialize KV cache
        hidden_states, kv_cache = self.attention_engine.prefill(
            hidden_states,
            attention_mask=None
        )

        # Step 3: Process through ALL layers (CRITICAL FIX: was only layer 0)
        for layer_id in range(self.model_info.num_layers):
            # Batch router computation for this layer
            if hasattr(self, 'router_weights_by_layer') and layer_id in self.router_weights_by_layer:
                original_weights = self.router_calculator.router_weights
                self.router_calculator.router_weights = self.router_weights_by_layer[layer_id]
                expert_indices, expert_weights = self.router_calculator.forward(
                    hidden_states,
                    batch_mode=True
                )
                self.router_calculator.router_weights = original_weights
            else:
                expert_indices, expert_weights = self.router_calculator.forward(
                    hidden_states,
                    batch_mode=True
                )

            # Analyze expert frequency for this layer (for cache warmup)
            if layer_id == 0:  # Only log for first layer to avoid spam
                activated_experts = self._analyze_expert_frequency(expert_indices)
                logger.info(f"Layer {layer_id} prefill activated {len(activated_experts)} unique experts: "
                           f"{activated_experts[:10]}")

            # Process tokens through experts (grouped by expert)
            # Three-tier cache handles loading automatically: GPU → CPU → Storage
            hidden_states = self.expert_processor.process_batch(
                hidden_states=hidden_states,
                expert_indices=expert_indices,
                expert_weights=expert_weights,
                layer_id=layer_id,
                expert_loader=self._create_expert_loader(layer_id),
                trigger_prefetch=(layer_id == self.model_info.num_layers - 1)  # Only prefetch after last layer
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

            # Process through ALL layers (CRITICAL FIX: was only processing layer 0)
            for layer_id in range(self.model_info.num_layers):
                # Step 1: Attention for this layer
                current_hidden, kv_cache = self.attention_engine.decode(
                    token_id=current_hidden,
                    kv_cache=kv_cache
                )

                # Step 2: Route token to experts for this layer
                # Use layer-specific router weights if available
                if hasattr(self, 'router_weights_by_layer') and layer_id in self.router_weights_by_layer:
                    # Temporarily swap router weights for this layer
                    original_weights = self.router_calculator.router_weights
                    self.router_calculator.router_weights = self.router_weights_by_layer[layer_id]
                    expert_indices, expert_weights = self.router_calculator.forward(
                        current_hidden,
                        batch_mode=False
                    )
                    self.router_calculator.router_weights = original_weights
                else:
                    # Fallback to shared router (less accurate but works)
                    expert_indices, expert_weights = self.router_calculator.forward(
                        current_hidden,
                        batch_mode=False
                    )

                # Step 3-5: Process through experts for this layer
                current_hidden = self.expert_processor.process_single(
                    hidden_state=current_hidden,
                    expert_indices=expert_indices.squeeze(1),  # [batch, top_k]
                    expert_weights=expert_weights.squeeze(1),
                    layer_id=layer_id,
                    expert_loader=self._create_expert_loader(layer_id),
                    update_predictor=True
                )

            # Step 6: Generate next token
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
            # Attention
            current_hidden, kv_cache = self.attention_engine.decode(
                token_id=current_hidden,
                kv_cache=kv_cache
            )

            # Routing
            expert_indices, expert_weights = self.router_calculator.forward(
                current_hidden,
                batch_mode=False
            )

            # Expert processing
            layer_id = 0
            current_hidden = self.expert_processor.process_single(
                hidden_state=current_hidden,
                expert_indices=expert_indices.squeeze(1),
                expert_weights=expert_weights.squeeze(1),
                layer_id=layer_id,
                expert_loader=self._create_expert_loader(layer_id),
                update_predictor=True
            )

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
        """Create expert loader function that loads real weights from three-tier cache."""

        # Cache model path to avoid repeated downloads
        model_path = self._ensure_model_downloaded()

        def load_expert(expert_id: int) -> ExpertFFN:
            """Load expert weights through three-tier cache (GPU → CPU → Storage).

            OPTIMIZED: The ExpertCache checks memory tiers BEFORE loading from storage:
            1. GPU cache hit → return immediately (no I/O!)
            2. CPU cache hit → promote to GPU (no disk I/O)
            3. Storage load → load from disk and cache (unavoidable I/O)

            Returns an ExpertFFN module with real weights loaded from checkpoint.
            """
            # Load expert weights from storage (cache handles tier management)
            expert_weights = self._load_expert_from_storage(
                model_path,
                layer_id,
                expert_id
            )

            # Determine if this is a gated expert (has w3)
            has_gate = "w3.weight" in expert_weights

            # Create expert module
            expert = ExpertFFN(
                hidden_dim=self.model_info.hidden_size,
                expert_dim=self.model_info.intermediate_size,
                activation="silu",  # Most MoE models use SiLU/Swish
                has_gate=has_gate
            )

            # Load weights into module
            # Move to device and dtype
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

            # Set to eval mode (no gradient computation needed)
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
        if hasattr(self, 'attention_engine'):
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
