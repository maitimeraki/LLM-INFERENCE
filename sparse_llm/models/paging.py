"""Paging capabilities and validation for MoE models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class PagingCapabilities:
    """Paging capabilities for a model."""

    supported: bool = False
    page_size: Optional[int] = None
    max_pages: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def from_mixtral(cls, metadata: Optional[Dict[str, Any]] = None) -> PagingCapabilities:
        """Create paging capabilities from Mixtral metadata."""
        return cls(
            supported=False,
            page_size=None,
            max_pages=None,
            metadata=metadata,
        )


@dataclass
class PagingValidationResult:
    """Result of paging validation."""

    valid: bool
    errors: list[str]
    warnings: list[str]

    @classmethod
    def success(cls) -> PagingValidationResult:
        """Create a successful validation result."""
        return cls(valid=True, errors=[], warnings=[])

    @classmethod
    def failure(cls, errors: list[str], warnings: Optional[list[str]] = None) -> PagingValidationResult:
        """Create a failed validation result."""
        return cls(valid=False, errors=errors, warnings=warnings or [])


__all__ = ["PagingCapabilities", "PagingValidationResult"]
