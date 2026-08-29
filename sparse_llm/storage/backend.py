"""Storage interfaces and the canonical local tensor backend."""

from __future__ import annotations

import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

import torch

from sparse_llm.cache.expert_cache import ExpertKey


class StorageBackend(ABC):
    """Interface for loading and persisting expert tensors."""

    @abstractmethod
    def load_expert(
        self, expert_id: ExpertKey | int, layer_id: int | None = None
    ) -> torch.Tensor:
        """Load an expert tensor onto the CPU."""

    @abstractmethod
    def save_expert(
        self,
        expert_id: ExpertKey | int,
        weights: torch.Tensor,
        layer_id: int | None = None,
    ) -> None:
        """Persist an expert tensor."""

    @abstractmethod
    def exists(self, expert_id: ExpertKey | int, layer_id: int | None = None) -> bool:
        """Return whether an expert tensor is persisted."""


class LocalSSDStorage(StorageBackend):
    """Persist expert tensors as atomically replaced PyTorch files."""

    def __init__(self, base_path: str | Path = "./cache/experts") -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_id(value: object, name: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
        return value

    @classmethod
    def _normalize_key(
        cls, expert_id: ExpertKey | int, layer_id: int | None = None
    ) -> ExpertKey | int:
        if isinstance(expert_id, tuple):
            if len(expert_id) != 2:
                raise ValueError("ExpertKey must be a (layer_id, expert_id) tuple")
            tuple_layer, tuple_expert = expert_id
            tuple_layer = cls._validate_id(tuple_layer, "layer_id")
            tuple_expert = cls._validate_id(tuple_expert, "expert_id")
            if layer_id is not None:
                layer_id = cls._validate_id(layer_id, "layer_id")
                if layer_id != tuple_layer:
                    raise ValueError("layer_id conflicts with the ExpertKey")
            return (tuple_layer, tuple_expert)

        expert_id = cls._validate_id(expert_id, "expert_id")
        if layer_id is None:
            return expert_id
        return (cls._validate_id(layer_id, "layer_id"), expert_id)

    def _get_path(
        self, expert_id: ExpertKey | int, layer_id: int | None = None
    ) -> Path:
        """Resolve a key to its deterministic on-disk path."""
        key = self._normalize_key(expert_id, layer_id)
        if isinstance(key, tuple):
            layer, expert = key
            filename = f"layer_{layer:04d}_expert_{expert:04d}.pt"
        else:
            filename = f"expert_{key:06d}.pt"
        return self.base_path / filename

    def _candidate_paths(self, key: ExpertKey | int) -> list[Path]:
        """Return lookup paths, resolving a bare integer only when unique."""
        if isinstance(key, tuple):
            return [self._get_path(key)]

        legacy_path = self._get_path(key)
        if legacy_path.exists():
            return [legacy_path]

        matches = sorted(self.base_path.glob(f"layer_*_expert_{key:04d}.pt"))
        if len(matches) > 1:
            raise ValueError(f"expert {key} has ambiguous layer-aware storage entries")
        return matches or [legacy_path]

    def _lookup_path(
        self, expert_id: ExpertKey | int, layer_id: int | None = None
    ) -> tuple[ExpertKey | int, Path]:
        key = self._normalize_key(expert_id, layer_id)
        if layer_id is not None or isinstance(expert_id, tuple):
            return key, self._get_path(key)
        return key, self._candidate_paths(key)[0]

    def load_expert(
        self, expert_id: ExpertKey | int, layer_id: int | None = None
    ) -> torch.Tensor:
        key, path = self._lookup_path(expert_id, layer_id)
        if not path.exists():
            if isinstance(key, tuple):
                layer, expert = key
                raise FileNotFoundError(
                    f"expert at layer {layer}, expert {expert} not found at {path}"
                )
            raise FileNotFoundError(f"expert {key} not found at {path}")
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def save_expert(
        self,
        expert_id: ExpertKey | int,
        weights: torch.Tensor,
        layer_id: int | None = None,
    ) -> None:
        path = self._get_path(expert_id, layer_id)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                dir=self.base_path,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                torch.save(weights.detach().cpu(), temporary_file)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def exists(self, expert_id: ExpertKey | int, layer_id: int | None = None) -> bool:
        key = self._normalize_key(expert_id, layer_id)
        paths = [self._get_path(key)] if layer_id is not None or isinstance(expert_id, tuple) else self._candidate_paths(key)
        return paths[0].exists()


__all__ = ["ExpertKey", "StorageBackend", "LocalSSDStorage"]
