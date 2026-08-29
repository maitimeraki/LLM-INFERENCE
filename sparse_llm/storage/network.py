import logging
from typing import Dict, Optional
import torch
import requests
import json
from sparse_llm.storage.base import StorageBackend, ExpertMetadata

logger = logging.getLogger(__name__)


class NetworkStorage(StorageBackend):
    """Fetch experts from remote HTTP server."""

    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip('/')
        self.experts_cache: Dict[int, ExpertMetadata] = {}
        self._fetch_expert_list()

    def _fetch_expert_list(self) -> None:
        """Fetch list of available experts from server."""
        try:
            response = requests.get(f"{self.server_url}/experts", timeout=5)
            response.raise_for_status()
            data = response.json()
            for exp_id, metadata in data.items():
                self.experts_cache[int(exp_id)] = ExpertMetadata(**metadata)
            logger.info(f"Fetched {len(self.experts_cache)} experts from {self.server_url}")
        except Exception as e:
            logger.warning(f"Failed to fetch expert list: {e}")

    def load_expert(self, expert_id: int) -> torch.Tensor:
        """Fetch expert weights from remote server."""
        try:
            response = requests.get(
                f"{self.server_url}/experts/{expert_id}",
                timeout=30
            )
            response.raise_for_status()
            weights = torch.frombuffer(response.content, dtype=torch.float32)
            logger.debug(f"Fetched expert {expert_id} from {self.server_url}")
            return weights
        except Exception as e:
            logger.error(f"Failed to fetch expert {expert_id}: {e}")
            raise

    def save_expert(self, expert_id: int, weights: torch.Tensor, metadata: ExpertMetadata) -> None:
        """Network storage is read-only."""
        raise NotImplementedError("Network storage backend is read-only")

    def exists(self, expert_id: int) -> bool:
        """Check if expert exists on remote server."""
        return expert_id in self.experts_cache

    def list_experts(self) -> Dict[int, ExpertMetadata]:
        """List all experts on remote server."""
        return self.experts_cache.copy()

    def delete_expert(self, expert_id: int) -> None:
        """Network storage is read-only."""
        raise NotImplementedError("Network storage backend is read-only")
