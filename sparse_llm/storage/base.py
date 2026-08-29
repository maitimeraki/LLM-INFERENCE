import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict
import torch

logger = logging.getLogger(__name__)


@dataclass
class ExpertMetadata:
    expert_id: int
    layer_idx: int
    size_bytes: int
    dtype: str
    quantized: bool
    checksum: Optional[str] = None


class StorageBackend(ABC):
    """Abstract base class for expert storage backends."""

    @abstractmethod
    def load_expert(self, expert_id: int) -> torch.Tensor:
        """Load expert weights from storage."""
        pass

    @abstractmethod
    def save_expert(self, expert_id: int, weights: torch.Tensor, metadata: ExpertMetadata) -> None:
        """Save expert weights to storage."""
        pass

    @abstractmethod
    def exists(self, expert_id: int) -> bool:
        """Check if expert exists in storage."""
        pass

    @abstractmethod
    def list_experts(self) -> Dict[int, ExpertMetadata]:
        """List all available experts."""
        pass

    @abstractmethod
    def delete_expert(self, expert_id: int) -> None:
        """Delete expert from storage."""
        pass
