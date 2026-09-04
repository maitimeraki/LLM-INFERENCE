"""SparseLLM vLLM Integration Package

This package provides the integration layer between SparseLLM's expert paging
system and vLLM's serving infrastructure.
"""

from sparse_llm.integrations.vllm_plugin import (
    VLLMSparseExpertPager,
    VLLMSparseMoELayer,
    replace_moe_layers_with_paged,
    VLLM_AVAILABLE,
)

__all__ = [
    "VLLMSparseExpertPager",
    "VLLMSparseMoELayer",
    "replace_moe_layers_with_paged",
    "VLLM_AVAILABLE",
]
