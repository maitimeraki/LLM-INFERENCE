"""Architecture-neutral contracts for validated expert paging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import torch


ExpertKey = tuple[int, int]
ExpertForward = Callable[[ExpertKey, torch.Tensor, "LoadedExpertProtocol"], torch.Tensor]
Router = Callable[[torch.Tensor], "RouterSelection"]


class LoadedExpertProtocol:
    """Minimal protocol-like shape consumed by architecture adapters."""

    tensors: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class TensorSchema:
    """Shape, dtype, and byte contract for one expert tensor."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    byte_size: int

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("tensor schema name must be non-empty")
        if any(not isinstance(dimension, int) or dimension < 0 for dimension in self.shape):
            raise ValueError("tensor schema dimensions must be non-negative integers")
        if not self.dtype.strip():
            raise ValueError("tensor schema dtype must be non-empty")
        if not isinstance(self.byte_size, int) or isinstance(self.byte_size, bool) or self.byte_size < 0:
            raise ValueError("tensor schema byte_size must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "byte_size": self.byte_size,
        }


@dataclass(frozen=True)
class RouterSelection:
    """The authoritative top-k router result for one MoE layer."""

    expert_indices: torch.Tensor
    routing_probabilities: torch.Tensor

    def validate(self, *, token_count: int, num_experts: int, top_k: int) -> None:
        if self.expert_indices.ndim != 2 or self.routing_probabilities.ndim != 2:
            raise ValueError("router outputs must be rank-2 [tokens, top_k] tensors")
        if self.expert_indices.shape != self.routing_probabilities.shape:
            raise ValueError("router indices and probabilities must have equal shapes")
        if self.expert_indices.shape != (token_count, top_k):
            raise ValueError(
                f"router output shape {tuple(self.expert_indices.shape)} does not match "
                f"({token_count}, {top_k})"
            )
        if self.expert_indices.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("router expert indices must use an integer dtype")
        if self.expert_indices.numel():
            minimum = int(self.expert_indices.min().item())
            maximum = int(self.expert_indices.max().item())
            if minimum < 0 or maximum >= num_experts:
                raise ValueError(
                    f"router expert index range [{minimum}, {maximum}] is outside "
                    f"0..{num_experts - 1}"
                )
        if not torch.isfinite(self.routing_probabilities).all():
            raise ValueError("router probabilities must be finite")


@dataclass(frozen=True)
class PagingCapabilities:
    """Evidence-backed capabilities of one validated architecture adapter."""

    architecture_id: str
    adapter_version: str
    num_layers: int
    num_experts: int
    top_k: int
    router_output_shape: tuple[int, ...] | None = None
    shared_tensor_groups: tuple[str, ...] = ()
    expert_tensor_names: tuple[str, ...] = ()
    expert_tensor_schema: tuple[TensorSchema, ...] = ()
    supports_prefill: bool = False
    supports_decode: bool = False
    supports_quantized_experts: bool = False
    reference_output_tolerance: float | None = None
    validation_passed: bool = False
    failure_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.architecture_id.strip():
            raise ValueError("architecture_id must be non-empty")
        if not self.adapter_version.strip():
            raise ValueError("adapter_version must be non-empty")
        for name, value in (
            ("num_layers", self.num_layers),
            ("num_experts", self.num_experts),
            ("top_k", self.top_k),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.top_k > self.num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        if self.reference_output_tolerance is not None and self.reference_output_tolerance < 0:
            raise ValueError("reference_output_tolerance must be non-negative")
        schema_names = {schema.name for schema in self.expert_tensor_schema}
        if len(schema_names) != len(self.expert_tensor_schema):
            raise ValueError("expert tensor schema names must be unique")
        if self.validation_passed and self.failure_reasons:
            raise ValueError("a passed paging capability cannot contain failure reasons")

    @classmethod
    def from_mixtral(cls, metadata: Any) -> "PagingCapabilities":
        """Factory method to create PagingCapabilities from Mixtral paging metadata.

        Args:
            metadata: MixtralPagingMetadata with num_hidden_layers, num_local_experts,
                     and num_experts_per_tok attributes.

        Returns:
            PagingCapabilities configured for Mixtral expert paging.
        """
        return cls(
            architecture_id="mixtral-8x7b",
            adapter_version="1.0",
            num_layers=metadata.num_hidden_layers,
            num_experts=metadata.num_local_experts,
            top_k=metadata.num_experts_per_tok,
            expert_tensor_names=metadata.expert_tensor_names,
            supports_prefill=True,
            supports_decode=True,
            validation_passed=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture_id": self.architecture_id,
            "adapter_version": self.adapter_version,
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "router_output_shape": self.router_output_shape,
            "shared_tensor_groups": list(self.shared_tensor_groups),
            "expert_tensor_names": list(self.expert_tensor_names),
            "expert_tensor_schema": [schema.to_dict() for schema in self.expert_tensor_schema],
            "supports_prefill": self.supports_prefill,
            "supports_decode": self.supports_decode,
            "supports_quantized_experts": self.supports_quantized_experts,
            "reference_output_tolerance": self.reference_output_tolerance,
            "validation_passed": self.validation_passed,
            "failure_reasons": list(self.failure_reasons),
        }


@dataclass(frozen=True)
class PagingValidationResult:
    """Structured result explaining why paging is or is not eligible."""

    eligible: bool
    failure_reasons: tuple[str, ...] = ()

    @classmethod
    def from_capabilities(cls, capabilities: PagingCapabilities) -> "PagingValidationResult":
        return cls(capabilities.validation_passed, capabilities.failure_reasons)

    @classmethod
    def unavailable(cls, reason: str) -> "PagingValidationResult":
        if not reason.strip():
            raise ValueError("paging validation reason must be non-empty")
        return cls(False, (reason,))

    def to_dict(self) -> dict[str, Any]:
        return {"eligible": self.eligible, "failure_reasons": list(self.failure_reasons)}


@runtime_checkable
class ExpertPagingAdapter(Protocol):
    """Optional adapter boundary for architecture-validated paging."""

    @property
    def paging_capabilities(self) -> PagingCapabilities | None:
        ...

    def validate_paging(self) -> PagingValidationResult:
        ...


class PagedMoELayer:
    """Execute a validated layer using its real router and expert callback.

    The layer is deliberately small: architecture adapters provide the original
    router and expert computation, while this class owns only grouping, leases,
    routing-weight combination, and failure-safe release.
    """

    def __init__(
        self,
        *,
        router: Router,
        pager: Any,
        layer_index: int,
        num_experts: int,
        top_k: int,
        expert_forward: ExpertForward,
    ) -> None:
        if not callable(router) or not callable(expert_forward):
            raise TypeError("router and expert_forward must be callable")
        if not isinstance(layer_index, int) or layer_index < 0:
            raise ValueError("layer_index must be non-negative")
        if not isinstance(num_experts, int) or num_experts < 1:
            raise ValueError("num_experts must be positive")
        if not isinstance(top_k, int) or top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must be in the range 1..num_experts")
        self.router = router
        self.pager = pager
        self.layer_index = layer_index
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_forward = expert_forward

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim < 2 or hidden.shape[-1] < 1:
            raise ValueError("hidden states must have shape [..., tokens, hidden_size]")
        flat = hidden.reshape(-1, hidden.shape[-1])
        selection = self.router(flat)
        if not isinstance(selection, RouterSelection):
            raise TypeError("router must return RouterSelection")
        selection.validate(token_count=flat.shape[0], num_experts=self.num_experts, top_k=self.top_k)
        indices = selection.expert_indices.to(device=flat.device)
        probabilities = selection.routing_probabilities.to(device=flat.device)
        requested = sorted({int(expert) for expert in indices.flatten().tolist()})
        record = getattr(self.pager, "record_routed", None)
        if callable(record):
            record((self.layer_index, expert) for expert in requested)

        assignments: dict[int, list[tuple[int, int]]] = {expert: [] for expert in requested}
        for token_index in range(indices.shape[0]):
            for choice in range(self.top_k):
                assignments[int(indices[token_index, choice])].append((token_index, choice))

        output: torch.Tensor | None = None
        for expert_id in requested:
            assignments_for_expert = assignments[expert_id]
            unique_positions = sorted({token for token, _ in assignments_for_expert})
            position_tensor = torch.tensor(unique_positions, device=flat.device, dtype=torch.long)
            with self.pager.lease((self.layer_index, expert_id)) as loaded:
                expert_output = self.expert_forward(
                    (self.layer_index, expert_id), flat.index_select(0, position_tensor), loaded
                )
            if expert_output.ndim != 2 or expert_output.shape[0] != len(unique_positions):
                raise ValueError("expert output must be [selected_tokens, output_size]")
            if output is None:
                output = torch.zeros(
                    (flat.shape[0], expert_output.shape[-1]),
                    dtype=expert_output.dtype,
                    device=expert_output.device,
                )
            if output.device != expert_output.device or output.dtype != expert_output.dtype:
                raise ValueError("expert outputs must share device and dtype")
            lookup = {position: offset for offset, position in enumerate(unique_positions)}
            for token_index, choice in assignments_for_expert:
                output[token_index] += probabilities[token_index, choice] * expert_output[
                    lookup[token_index]
                ]
        if output is None:
            raise ValueError("router selected no experts")
        return output.reshape(*hidden.shape[:-1], output.shape[-1])


__all__ = [
    "ExpertForward",
    "ExpertKey",
    "ExpertPagingAdapter",
    "LoadedExpertProtocol",
    "PagedMoELayer",
    "PagingCapabilities",
    "PagingValidationResult",
    "Router",
    "RouterSelection",
    "TensorSchema",
]
