import logging
import os
import tempfile
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class LocalSSDStorage:
    """Simple CPU tensor store for baseline expert offloading."""

    def __init__(self, base_path: str = "./cache/experts"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_path(self, expert_id: int, layer_id: int | None = None) -> Path:
        name = f"layer_{layer_id:04d}_expert_{expert_id:04d}.pt" if layer_id is not None else f"expert_{expert_id:06d}.pt"
        return self.base_path / name

    def load_expert(self, expert_id: int, layer_id: int | None = None) -> torch.Tensor:
        path = self._get_path(expert_id, layer_id)
        if not path.exists():
            raise FileNotFoundError(f"Expert {expert_id} (layer {layer_id}) not found at {path}")
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def save_expert(self, expert_id: int, weights: torch.Tensor, layer_id: int | None = None) -> None:
        path = self._get_path(expert_id, layer_id)
        with tempfile.NamedTemporaryFile(dir=self.base_path, delete=False) as tmp:
            temporary_path = Path(tmp.name)
        try:
            torch.save(weights.detach().cpu(), temporary_path)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def exists(self, expert_id: int, layer_id: int | None = None) -> bool:
        return self._get_path(expert_id, layer_id).exists()


__all__ = ["LocalSSDStorage"]
