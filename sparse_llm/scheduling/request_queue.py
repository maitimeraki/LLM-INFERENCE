import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import time
import uuid

logger = logging.getLogger(__name__)


class RequestStatus(Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class InferenceRequest:
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    user_id: Optional[str] = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    status: RequestStatus = RequestStatus.QUEUED
    result: Optional[str] = None
    error: Optional[str] = None
    processing_start_time: Optional[float] = None
    processing_end_time: Optional[float] = None

    def processing_time_ms(self) -> Optional[float]:
        if self.processing_start_time and self.processing_end_time:
            return (self.processing_end_time - self.processing_start_time) * 1000
        return None

    def queue_wait_time_ms(self) -> float:
        start = self.processing_start_time or time.time()
        return (start - self.created_at) * 1000


class RequestQueue:
    """FIFO queue for inference requests."""

    def __init__(self, max_queue_size: int = 1000):
        self.queue: List[InferenceRequest] = []
        self.max_queue_size = max_queue_size
        self.request_history: Dict[str, InferenceRequest] = {}

    def enqueue(self, request: InferenceRequest) -> bool:
        """Add request to queue. Returns False if queue full."""
        if len(self.queue) >= self.max_queue_size:
            logger.warning("Request queue full")
            return False
        self.queue.append(request)
        self.request_history[request.request_id] = request
        logger.debug(f"Enqueued request {request.request_id}")
        return True

    def dequeue(self) -> Optional[InferenceRequest]:
        """Pop next request from queue."""
        if self.queue:
            return self.queue.pop(0)
        return None

    def peek(self) -> Optional[InferenceRequest]:
        """View next request without removing."""
        return self.queue[0] if self.queue else None

    def get_status(self, request_id: str) -> Optional[RequestStatus]:
        """Get request status."""
        req = self.request_history.get(request_id)
        return req.status if req else None

    def size(self) -> int:
        """Current queue size."""
        return len(self.queue)

    def stats(self) -> Dict:
        """Return queue statistics."""
        completed = [r for r in self.request_history.values()
                    if r.status == RequestStatus.COMPLETED]
        avg_queue_wait = (sum(r.queue_wait_time_ms() for r in completed) / len(completed)
                         if completed else 0)
        avg_processing = (sum(r.processing_time_ms() for r in completed
                             if r.processing_time_ms()) / len(completed)
                         if completed else 0)
        return {
            "queue_size": self.size(),
            "total_processed": len(completed),
            "avg_queue_wait_ms": avg_queue_wait,
            "avg_processing_ms": avg_processing,
        }
