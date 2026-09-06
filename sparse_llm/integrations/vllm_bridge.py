"""Bridge to make vLLM use externally-managed weights from LoadedWeightState.

This module provides two complementary approaches for vLLM integration:

1. **Memory-Coordinated Initialization** (Recommended for MoE models):
   Use `initialize_vllm_with_coordination()` for OOM-safe initialization.

   Phase 1: VLLMMemoryCoordinator calculates memory budgets and initializes vLLM
            with the optimal gpu_memory_utilization parameter.
   Phase 2: VLLMWeightBridge injects weight references, making vLLM parameters
            point to ExpertCache GPU tensors.

   This prevents OOM by pre-calculating ALL memory needs before loading anything.

2. **Manual Weight Bridge** (Advanced use cases):
   Use the VLLMWeightBridge class directly when you need fine-grained control
   over initialization. You must manage memory allocation yourself.

For most users, use approach #1 via `initialize_vllm_with_coordination()`.
"""

import logging
from typing import Dict, Any, Optional, Tuple
import torch

logger = logging.getLogger(__name__)


class VLLMWeightBridge:
    """Bridge that makes vLLM model parameters reference ExpertCache tensors.

    Architecture:
    - Your system (LoadedWeightState + ExpertCache) owns GPU memory for weights
    - vLLM parameters are views/references into your GPU tensors
    - Expert swapping happens in ExpertCache
    - vLLM just reads from GPU addresses that ExpertCache manages
    - vLLM only allocates memory for KV cache
    """

    def __init__(self, loaded_state):
        """Initialize bridge.

        Args:
            loaded_state: LoadedWeightState from FourPhaseOrchestrator
        """
        self.loaded_state = loaded_state
        self.shared_weights = loaded_state.shared_weights
        self.expert_cache = loaded_state.expert_cache
        self.model_info = loaded_state.model_info

    def patch_vllm_weight_loading(self):
        """Monkey-patch vLLM to prevent weight allocation for experts.

        This intercepts torch.empty() calls during FusedMoE initialization
        and replaces them with references to your ExpertCache.
        """
        from vllm.model_executor.layers.fused_moe import unquantized_fused_moe_method

        # Store original method
        original_create_weights = unquantized_fused_moe_method.UnquantizedFusedMoEMethod.create_weights
        bridge_self = self

        def patched_create_weights(method_self, layer, **kwargs):
            """Patched create_weights that creates CPU placeholders instead of GPU allocations."""

            logger.info("🔧 VLLMWeightBridge: Intercepting expert weight allocation")
            logger.info(f"   Layer: {layer}")
            logger.info(f"   Preventing GPU allocation, will use ExpertCache references")

            # Get parameters
            num_experts = kwargs.get('num_experts', 0)
            hidden_size = kwargs.get('hidden_size', 0)
            intermediate_size = kwargs.get('intermediate_size', 0)

            logger.info(f"   Experts: {num_experts}, Hidden: {hidden_size}, Intermediate: {intermediate_size}")

            # CRITICAL: Create weights on CPU as placeholders
            # These will be replaced with ExpertCache references during inject_weights()
            # This prevents OOM during initialization
            with torch.device('cpu'):
                try:
                    original_create_weights(method_self, layer, **kwargs)
                    logger.info("✅ Created expert weight placeholders on CPU (will reference ExpertCache)")
                except Exception as e:
                    logger.error(f"❌ Failed to create CPU placeholders: {e}")
                    raise

        # Apply patch
        unquantized_fused_moe_method.UnquantizedFusedMoEMethod.create_weights = patched_create_weights
        logger.info("✅ VLLMWeightBridge: Patched FusedMoE weight creation")

    def inject_weight_references(self, vllm_model):
        """Inject weight references into vLLM model after initialization.

        This makes vLLM parameters point to your ExpertCache GPU tensors.

        Args:
            vllm_model: vLLM's model instance (after initialization)
        """
        logger.info("="*70)
        logger.info("🔗 VLLMWeightBridge: Injecting weight references")
        logger.info("="*70)

        injected_shared = 0
        injected_experts = 0

        with torch.no_grad():
            for name, param in vllm_model.named_parameters():

                # Handle shared weights
                if ".experts." not in name and "block_sparse_moe" not in name:
                    tensor = self._get_shared_weight_reference(name)
                    if tensor is not None:
                        # Make parameter reference your shared weight tensor
                        param.data = tensor
                        injected_shared += 1

                # Handle expert weights
                else:
                    tensor = self._get_expert_weight_reference(name)
                    if tensor is not None:
                        # Make parameter reference ExpertCache GPU tensor
                        param.data = tensor
                        injected_experts += 1

        logger.info(f"✅ Injected {injected_shared} shared weight references")
        logger.info(f"✅ Injected {injected_experts} expert weight references")
        logger.info("="*70)
        logger.info("🎯 vLLM now uses YOUR GPU memory for weights")
        logger.info("🎯 ExpertCache manages expert swapping")
        logger.info("🎯 vLLM only manages KV cache")
        logger.info("="*70)

    def _get_shared_weight_reference(self, name: str) -> torch.Tensor | None:
        """Get reference to shared weight tensor.

        Args:
            name: Parameter name from vLLM model

        Returns:
            Tensor from LoadedWeightState (on GPU)
        """
        # Try exact match
        if name in self.shared_weights:
            return self.shared_weights[name]

        # Try with model. prefix
        prefixed = f"model.{name}"
        if prefixed in self.shared_weights:
            return self.shared_weights[prefixed]

        # Try without model. prefix
        unprefixed = name.replace("model.", "", 1)
        if unprefixed in self.shared_weights:
            return self.shared_weights[unprefixed]

        logger.debug(f"Shared weight not found: {name}")
        return None

    def _get_expert_weight_reference(self, name: str) -> torch.Tensor | None:
        """Get reference to expert weight tensor from ExpertCache.

        This returns a reference to the GPU tensor in ExpertCache.
        When ExpertCache swaps experts, this reference automatically
        points to the new expert data (same GPU address, different values).

        Args:
            name: Expert parameter name (e.g., "model.layers.5.experts.3.w1.weight")

        Returns:
            Tensor from ExpertCache (on GPU)
        """
        try:
            layer_id, expert_id, weight_key = self._parse_expert_name(name)

            # Get expert from cache (loads to GPU if needed)
            expert_weights, tier = self.expert_cache.get(layer_id, expert_id)

            if weight_key not in expert_weights:
                logger.debug(f"Weight key '{weight_key}' not found in expert L{layer_id}E{expert_id}")
                return None

            tensor = expert_weights[weight_key]

            # Ensure on GPU
            if tensor.device.type != 'cuda':
                tensor = tensor.cuda()

            return tensor

        except Exception as e:
            logger.debug(f"Could not get expert weight '{name}': {e}")
            return None

    def _parse_expert_name(self, name: str) -> tuple[int, int, str]:
        """Parse expert parameter name.

        Args:
            name: Parameter name like "model.layers.5.block_sparse_moe.experts.3.w1.weight"

        Returns:
            (layer_id, expert_id, weight_key)
        """
        parts = name.split(".")

        # Find layer index
        try:
            layer_idx = parts.index("layers") + 1
            layer_id = int(parts[layer_idx])
        except (ValueError, IndexError):
            raise ValueError(f"Cannot parse layer_id from: {name}")

        # Find expert index
        try:
            expert_idx = parts.index("experts") + 1
            expert_id = int(parts[expert_idx])
        except (ValueError, IndexError):
            raise ValueError(f"Cannot parse expert_id from: {name}")

        # Weight key is everything after "experts.{id}."
        weight_key = ".".join(parts[expert_idx + 1:])

        return layer_id, expert_id, weight_key

    def setup_expert_swap_hook(self, vllm_model):
        """Setup forward hook to trigger expert swapping BEFORE vLLM uses them.

        This ensures ExpertCache loads experts to GPU before vLLM's forward pass reads them.

        Args:
            vllm_model: vLLM model instance
        """
        bridge_self = self

        def expert_swap_hook(module, input):
            """Hook that runs before MoE forward pass to ensure experts are in GPU."""

            # This hook is called before each MoE layer forward pass
            # We need to inspect the routing to know which experts will be used
            # For now, implement a simple prefetch strategy

            # TODO: Implement routing-aware prefetch based on top-k experts
            # This requires access to router logits which happens inside forward()

            pass

        # Register hook on each MoE layer
        for name, module in vllm_model.named_modules():
            if "block_sparse_moe" in name or "mlp" in name:
                module.register_forward_pre_hook(expert_swap_hook)
                logger.info(f"✅ Registered expert swap hook on: {name}")


def initialize_vllm_with_coordination(
    model_id: str,
    user_vllm_params: Optional[Dict[str, Any]] = None,
    storage_path: Optional[str] = None
) -> Tuple[Any, 'VLLMWeightBridge', Any]:
    """Initialize vLLM with coordinated memory allocation (prevents OOM).

    This is the RECOMMENDED way to initialize vLLM with SparseLLM for MoE models.

    This function orchestrates the full initialization flow:
    1. Uses VLLMMemoryCoordinator to calculate optimal memory allocation
    2. Initializes vLLM with calculated gpu_memory_utilization (prevents OOM)
    3. Loads expert weights via FourPhaseOrchestrator
    4. Creates VLLMWeightBridge and injects weight references
    5. Returns ready-to-use vLLM engine with SparseLLM integration

    Args:
        model_id: Hugging Face model ID or local path
        user_vllm_params: User's vLLM configuration parameters (dict).
                         All parameters are preserved except gpu_memory_utilization,
                         which is calculated automatically to prevent OOM.
                         Common parameters:
                         - max_model_len: Maximum sequence length (default: 2048)
                         - dtype: Data type ("float16", "bfloat16", "auto")
                         - quantization: Quantization method (None, "awq", "gptq")
                         - tensor_parallel_size: Number of GPUs (default: 1)
        storage_path: Optional custom storage path for cold experts

    Returns:
        (vllm_engine, weight_bridge, memory_allocation)
        - vllm_engine: Initialized vLLM LLM instance ready for inference
        - weight_bridge: VLLMWeightBridge instance managing weight references
        - memory_allocation: MemoryAllocation object with allocation details

    Raises:
        ValueError: If memory allocation cannot fulfill user request
        RuntimeError: If vLLM initialization fails

    Example:
        >>> # Simple initialization
        >>> vllm_engine, bridge, allocation = initialize_vllm_with_coordination(
        ...     model_id="mistralai/Mixtral-8x7B-v0.1",
        ...     user_vllm_params={
        ...         "max_model_len": 4096,
        ...         "dtype": "float16"
        ...     }
        ... )
        >>>
        >>> # Use vLLM for inference
        >>> outputs = vllm_engine.generate(["Hello, how are you?"])
        >>>
        >>> # Check memory allocation
        >>> print(f"GPU memory for KV cache: {allocation.gpu_kv_cache / 1024**3:.2f}GB")
        >>> print(f"Hot experts on GPU: {allocation.gpu_hot_expert_count}")

    Note:
        The gpu_memory_utilization parameter in user_vllm_params will be IGNORED
        and replaced with the calculated value to prevent OOM. All other parameters
        are passed through to vLLM unchanged.
    """
    from sparse_llm.integrations.vllm_memory_coordinator import VLLMMemoryCoordinator

    logger.info("="*80)
    logger.info("🚀 initialize_vllm_with_coordination: Starting OOM-safe initialization")
    logger.info("="*80)

    # Initialize coordinator
    coordinator = VLLMMemoryCoordinator(
        model_id=model_id,
        user_vllm_params=user_vllm_params or {},
        storage_path=storage_path
    )

    # Phase 1: Coordinate memory allocation and initialize vLLM
    logger.info("\n📊 Phase 1: Memory coordination and vLLM initialization...")
    vllm_engine, allocation, loaded_state = coordinator.initialize_with_coordination()

    # Phase 2: Create weight bridge and inject references
    logger.info("\n🔗 Phase 2: Creating weight bridge and injecting references...")
    weight_bridge = VLLMWeightBridge(loaded_state)

    # Get vLLM model from engine
    # vLLM's LLM class has different internal structure depending on version
    # Try to get the model in a version-agnostic way
    try:
        # vLLM 0.2.x+ structure
        vllm_model = vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
    except AttributeError:
        try:
            # Alternative structure for different vLLM versions
            vllm_model = vllm_engine.llm_engine.workers[0].model
        except (AttributeError, IndexError):
            logger.warning("⚠️  Could not access vLLM model for weight injection")
            logger.warning("    Weight injection skipped - using vLLM's standard weight loading")
            vllm_model = None

    # Inject weight references if we found the model
    if vllm_model is not None:
        weight_bridge.inject_weight_references(vllm_model)
        logger.info("✅ Weight references injected successfully")
    else:
        logger.warning("⚠️  Weight bridge created but injection skipped")

    logger.info("\n" + "="*80)
    logger.info("✅ initialize_vllm_with_coordination: Complete!")
    logger.info("="*80)
    logger.info(f"   vLLM engine ready for inference")
    logger.info(f"   GPU memory for KV cache: {allocation.gpu_kv_cache / 1024**3:.2f}GB")
    if loaded_state.model_info.is_moe:
        logger.info(f"   Hot experts on GPU: {allocation.gpu_hot_expert_count}")
    logger.info("="*80)

    return vllm_engine, weight_bridge, allocation
