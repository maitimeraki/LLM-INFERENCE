"""Conservative device-memory admission for paged inference."""

from __future__ import annotations

from dataclasses import dataclass


class MemoryAdmissionError(MemoryError):
    """Raised when a request cannot fit without changing its semantics."""


@dataclass(frozen=True)
class MemoryPlan:
    """Measured or configured memory accounting for one request."""

    capacity_bytes: int
    shared_bytes: int
    current_expert_bytes: int
    routed_group_bytes: int
    kv_cache_bytes: int
    activation_bytes: int
    overhead_bytes: int
    safety_margin_bytes: int
    configured_expert_budget_bytes: int | None
    reserved_non_expert_bytes: int
    available_expert_bytes: int
    can_admit: bool
    rejection_reason: str | None = None

    def to_dict(self) -> dict[str, int | bool | str | None]:
        return {
            "capacity_bytes": self.capacity_bytes,
            "shared_bytes": self.shared_bytes,
            "current_expert_bytes": self.current_expert_bytes,
            "routed_group_bytes": self.routed_group_bytes,
            "kv_cache_bytes": self.kv_cache_bytes,
            "activation_bytes": self.activation_bytes,
            "overhead_bytes": self.overhead_bytes,
            "safety_margin_bytes": self.safety_margin_bytes,
            "configured_expert_budget_bytes": self.configured_expert_budget_bytes,
            "reserved_non_expert_bytes": self.reserved_non_expert_bytes,
            "available_expert_bytes": self.available_expert_bytes,
            "can_admit": self.can_admit,
            "rejection_reason": self.rejection_reason,
        }


class MemoryPlanner:
    """Calculate a conservative byte budget before admitting expert tensors."""

    def __init__(
        self,
        *,
        capacity_bytes: int,
        expert_cache_bytes: int | None = None,
        kv_cache_bytes: int = 0,
        activation_bytes: int = 0,
        overhead_bytes: int = 0,
        safety_margin_bytes: int = 0,
    ) -> None:
        values = {
            "capacity_bytes": capacity_bytes,
            "expert_cache_bytes": expert_cache_bytes,
            "kv_cache_bytes": kv_cache_bytes,
            "activation_bytes": activation_bytes,
            "overhead_bytes": overhead_bytes,
            "safety_margin_bytes": safety_margin_bytes,
        }
        for name, value in values.items():
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")
        if capacity_bytes < 1:
            raise ValueError("capacity_bytes must be positive")
        if expert_cache_bytes == 0:
            raise ValueError("expert_cache_bytes must be positive when supplied")
        self.capacity_bytes = capacity_bytes
        self.expert_cache_bytes = expert_cache_bytes
        self.kv_cache_bytes = kv_cache_bytes
        self.activation_bytes = activation_bytes
        self.overhead_bytes = overhead_bytes
        self.safety_margin_bytes = safety_margin_bytes

    def plan(
        self,
        *,
        shared_bytes: int,
        routed_group_bytes: int = 0,
        current_expert_bytes: int = 0,
        kv_cache_bytes: int | None = None,
        activation_bytes: int | None = None,
        overhead_bytes: int | None = None,
    ) -> MemoryPlan:
        values = {
            "shared_bytes": shared_bytes,
            "routed_group_bytes": routed_group_bytes,
            "current_expert_bytes": current_expert_bytes,
            "kv_cache_bytes": self.kv_cache_bytes if kv_cache_bytes is None else kv_cache_bytes,
            "activation_bytes": self.activation_bytes if activation_bytes is None else activation_bytes,
            "overhead_bytes": self.overhead_bytes if overhead_bytes is None else overhead_bytes,
        }
        for name, value in values.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        reserved = (
            shared_bytes
            + values["kv_cache_bytes"]
            + values["activation_bytes"]
            + values["overhead_bytes"]
            + self.safety_margin_bytes
        )
        device_available = self.capacity_bytes - reserved
        available_expert = max(0, device_available)
        if self.expert_cache_bytes is not None:
            available_expert = min(available_expert, self.expert_cache_bytes)
        required_expert = current_expert_bytes + routed_group_bytes
        reason: str | None = None
        if reserved > self.capacity_bytes:
            reason = (
                f"resident shared/non-expert footprint is {reserved} bytes, exceeding "
                f"device capacity {self.capacity_bytes} bytes"
            )
        elif routed_group_bytes > available_expert:
            reason = (
                f"routed expert group requires {routed_group_bytes} bytes, but only "
                f"{available_expert} expert bytes are available"
            )
        elif required_expert > available_expert:
            reason = (
                f"current and routed experts require {required_expert} bytes, but only "
                f"{available_expert} expert bytes are available"
            )
        return MemoryPlan(
            capacity_bytes=self.capacity_bytes,
            shared_bytes=shared_bytes,
            current_expert_bytes=current_expert_bytes,
            routed_group_bytes=routed_group_bytes,
            kv_cache_bytes=values["kv_cache_bytes"],
            activation_bytes=values["activation_bytes"],
            overhead_bytes=values["overhead_bytes"],
            safety_margin_bytes=self.safety_margin_bytes,
            configured_expert_budget_bytes=self.expert_cache_bytes,
            reserved_non_expert_bytes=reserved,
            available_expert_bytes=available_expert,
            can_admit=reason is None,
            rejection_reason=reason,
        )

    @staticmethod
    def require(plan: MemoryPlan) -> MemoryPlan:
        if not plan.can_admit:
            raise MemoryAdmissionError(plan.rejection_reason or "memory admission rejected")
        return plan


__all__ = ["MemoryAdmissionError", "MemoryPlan", "MemoryPlanner"]
