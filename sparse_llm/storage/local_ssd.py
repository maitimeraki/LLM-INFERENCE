import logging
from pathlib import Path
from typing import Dict, Optional
import torch
import json
from sparse_llm.storage.base import StorageBackend, ExpertMetadata

logger = logging.getLogger(__name__)


class LocalSSDStorage(StorageBackend):
    """Store experts on local SSD with metadata."""

    def __init__(self, cache_dir: str = "./cache/experts"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata: Dict[int, ExpertMetadata] = self._load_metadata()

    def _load_metadata(self) -> Dict[int, ExpertMetadata]:
        """Load metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    return {int(k): ExpertMetadata(**v) for k, v in data.items()}
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
        return {}

    def _save_metadata(self) -> None:
        """Save metadata to disk."""
        try:
            data = {str(k): {
                'expert_id': v.expert_id,
                'layer_idx': v.layer_idx,
                'size_bytes': v.size_bytes,
                'dtype': v.dtype,
                'quantized': v.quantized,
                'checksum': v.checksum,
            } for k, v in self.metadata.items()}
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def _get_expert_path(self, expert_id: int) -> Path:
        return self.cache_dir / f"expert_{expert_id}.pt"

    def load_expert(self, expert_id: int) -> torch.Tensor:
        """Load expert weights from disk."""
        path = self._get_expert_path(expert_id)
        if not path.exists():
            raise FileNotFoundError(f"Expert {expert_id} not found at {path}")
        try:
            weights = torch.load(path, map_location='cpu')
            logger.debug(f"Loaded expert {expert_id} from {path}")
            return weights
        except Exception as e:
            logger.error(f"Failed to load expert {expert_id}: {e}")
            raise

    def save_expert(self, expert_id: int, weights: torch.Tensor, metadata: ExpertMetadata) -> None:
        """Save expert weights to disk."""
        path = self._get_expert_path(expert_id)
        try:
            torch.save(weights, path)
            self.metadata[expert_id] = metadata
            self._save_metadata()
            logger.info(f"Saved expert {expert_id} to {path}")
        except Exception as e:
            logger.error(f"Failed to save expert {expert_id}: {e}")
            raise

    def exists(self, expert_id: int) -> bool:
        """Check if expert exists on disk."""
        return self._get_expert_path(expert_id).exists()

    def list_experts(self) -> Dict[int, ExpertMetadata]:
        """List all experts in storage."""
        return self.metadata.copy()

    def delete_expert(self, expert_id: int) -> None:
        """Delete expert from disk."""
        path = self._get_expert_path(expert_id)
        try:
            if path.exists():
                path.unlink()
            self.metadata.pop(expert_id, None)
            self._save_metadata()
            logger.info(f"Deleted expert {expert_id}")
        except Exception as e:
            logger.error(f"Failed to delete expert {expert_id}: {e}")
            raise
