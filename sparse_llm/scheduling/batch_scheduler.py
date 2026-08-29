import logging
from typing import List, Dict, Optional
import time
from sparse_llm.scheduling.request_queue import InferenceRequest, RequestStatus, RequestQueue

logger = logging.getLogger(__name__)


class BatchScheduler:
    """Schedules requests into batches for inference."""

    def __init__(self, max_batch_size: int = 16, batch_timeout_ms: int = 100):
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.current_batch: List[InferenceRequest] = []
        self.batch_start_time: Optional[float] = None

    def add_request(self, request: InferenceRequest) -> bool:
        """Add request to current batch. Returns True if batch ready."""
        if len(self.current_batch) == 0:
            self.batch_start_time = time.time()

        self.current_batch.append(request)
        request.status = RequestStatus.PROCESSING
        request.processing_start_time = time.time()

        if len(self.current_batch) >= self.max_batch_size:
            return True
        return False

    def should_dispatch(self) -> bool:
        """Check if current batch should be dispatched."""
        if not self.current_batch:
            return False
        if len(self.current_batch) >= self.max_batch_size:
            return True
        if self.batch_start_time:
            elapsed_ms = (time.time() - self.batch_start_time) * 1000
            return elapsed_ms > self.batch_timeout_ms
        return False

    def get_batch(self) -> List[InferenceRequest]:
        """Get current batch for inference."""
        batch = self.current_batch
        self.current_batch = []
        self.batch_start_time = None
        return batch

    def batch_size(self) -> int:
        """Current batch size."""
        return len(self.current_batch)

    def stats(self) -> Dict:
        """Return scheduler statistics."""
        return {
            "current_batch_size": self.batch_size(),
            "max_batch_size": self.max_batch_size,
            "batch_timeout_ms": self.batch_timeout_ms,
        }
