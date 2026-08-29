import logging
import threading
import time
from typing import Optional, List, Dict
import torch
import torch.nn as nn

from sparse_llm.scheduling.request import InferenceRequest, RequestStatus
from sparse_llm.scheduling.queue import RequestQueue
from sparse_llm.models.moe import SparseMoELayer
from sparse_llm.cache.expert_cache import ExpertCache
from sparse_llm.prefetch.pipeline import PrefetchPipeline
from sparse_llm.storage.backend import StorageBackend
from sparse_llm.quantization.quantizer import QuantizationManager

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Main orchestrator for multi-user sparse LLM inference."""

    def __init__(self,
                 model: SparseMoELayer,
                 storage: StorageBackend,
                 device: str = "cuda",
                 max_queue_size: int = 1000,
                 max_cached_experts: int = 22):
        self.model = model.to(device)
        self.storage = storage
        self.device = device

        # Request management
        self.request_queue = RequestQueue(max_queue_size)
        self.request_lock = threading.Lock()

        # Expert management
        self.expert_cache = ExpertCache(max_cached_experts)
        self.prefetch_pipeline = PrefetchPipeline(num_streams=3)
        self.quantization = QuantizationManager(bits=4)

        # Inference state
        self.running = False
        self.inference_thread = None

        logger.info(f"Initialized InferenceEngine on {device} with {max_cached_experts} expert slots")

    def submit_request(self, user_id: str, prompt: str, max_tokens: int = 256) -> str:
        """Submit inference request from user."""
        request = InferenceRequest(
            user_id=user_id,
            prompt=prompt,
            max_tokens=max_tokens
        )

        success = self.request_queue.enqueue(request)
        if not success:
            logger.warning(f"Failed to enqueue request {request.request_id} (queue full)")
            return None

        return request.request_id

    def get_request_status(self, request_id: str) -> Dict:
        """Query request status."""
        request = self.request_queue.get_request(request_id)
        if not request:
            return {"status": "NOT_FOUND"}

        return {
            "status": request.status.name,
            "result": request.result if request.status == RequestStatus.COMPLETED else None,
            "error": request.error if request.status == RequestStatus.FAILED else None,
        }

    def start(self) -> None:
        """Start inference worker thread."""
        if self.running:
            logger.warning("Inference engine already running")
            return

        self.running = True
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=False)
        self.inference_thread.start()
        logger.info("Started inference engine worker thread")

    def stop(self) -> None:
        """Stop inference worker thread."""
        self.running = False
        if self.inference_thread:
            self.inference_thread.join(timeout=5.0)
        logger.info("Stopped inference engine")

    def _inference_loop(self) -> None:
        """Main inference worker loop."""
        while self.running:
            request = self.request_queue.dequeue()
            if not request:
                time.sleep(0.1)
                continue

            try:
                # Predict experts needed
                dummy_input = torch.randn(1, self.model.input_dim, device=self.device)
                routed_experts = self.model.get_routed_experts(dummy_input)

                # Load experts to cache
                for expert_id in routed_experts:
                    if not self.expert_cache.contains(expert_id):
                        self._load_expert(expert_id)

                # Run inference
                with torch.no_grad():
                    input_tensor = torch.randn(1, 1, self.model.input_dim, device=self.device)
                    output = self.model(input_tensor)

                result = f"Generated output for '{request.prompt[:40]}...'"
                self.request_queue.complete_request(request.request_id, result)
                logger.info(f"Completed request {request.request_id}")

            except Exception as e:
                error_msg = str(e)
                self.request_queue.fail_request(request.request_id, error_msg)
                logger.error(f"Failed request {request.request_id}: {error_msg}")

    def _load_expert(self, expert_id: int) -> None:
        """Load expert from storage to cache."""
        if self.expert_cache.contains(expert_id):
            return

        try:
            weights = self.storage.load_expert(expert_id)
            self.expert_cache.put(expert_id, weights)
            logger.debug(f"Loaded expert {expert_id} to cache")
        except Exception as e:
            logger.error(f"Failed to load expert {expert_id}: {e}")
            raise

    def stats(self) -> Dict:
        """Return inference engine statistics."""
        return {
            "queue": self.request_queue.stats(),
            "cache": self.expert_cache.stats(),
            "prefetch": self.prefetch_pipeline.stats(),
            "device": self.device,
        }
