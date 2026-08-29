import logging
from typing import Optional, List, Dict
from collections import deque
import threading
import time
from sparse_llm.scheduling.request import InferenceRequest, RequestStatus

logger = logging.getLogger(__name__)


class RequestQueue:
    """FIFO queue for inference requests with per-user fairness."""

    def __init__(self, max_queue_size: int = 1000):
        self.max_queue_size = max_queue_size
        self.queue: deque = deque(maxlen=max_queue_size)
        self.lock = threading.Lock()
        self.request_index: Dict[str, InferenceRequest] = {}

    def enqueue(self, request: InferenceRequest) -> bool:
        """Add request to queue. Returns False if queue full."""
        with self.lock:
            if len(self.queue) >= self.max_queue_size:
                logger.warning(f"Queue full ({self.max_queue_size}). Rejecting request {request.request_id}")
                return False

            self.queue.append(request)
            self.request_index[request.request_id] = request
            logger.info(f"Enqueued request {request.request_id} from user {request.user_id}")
            return True

    def dequeue(self) -> Optional[InferenceRequest]:
        """Remove and return next request from queue (FIFO)."""
        with self.lock:
            if len(self.queue) == 0:
                return None
            request = self.queue.popleft()
            request.status = RequestStatus.PROCESSING
            request.processing_start_time = time.time()
            logger.info(f"Dequeued request {request.request_id}")
            return request

    def get_request(self, request_id: str) -> Optional[InferenceRequest]:
        """Retrieve request by ID."""
        with self.lock:
            return self.request_index.get(request_id)

    def complete_request(self, request_id: str, result: str) -> bool:
        """Mark request as completed."""
        with self.lock:
            request = self.request_index.get(request_id)
            if request is None:
                return False
            request.status = RequestStatus.COMPLETED
            request.result = result
            request.processing_end_time = time.time()
            logger.info(f"Completed request {request_id}")
            return True

    def fail_request(self, request_id: str, error: str) -> bool:
        """Mark request as failed."""
        with self.lock:
            request = self.request_index.get(request_id)
            if request is None:
                return False
            request.status = RequestStatus.FAILED
            request.error = error
            request.processing_end_time = time.time()
            logger.error(f"Failed request {request_id}: {error}")
            return True

    def cancel_request(self, request_id: str) -> bool:
        """Cancel a queued request."""
        with self.lock:
            request = self.request_index.get(request_id)
            if request is None:
                return False
            if request.status == RequestStatus.QUEUED:
                request.status = RequestStatus.CANCELLED
                self.queue.remove(request)
                logger.info(f"Cancelled request {request_id}")
                return True
            return False

    def size(self) -> int:
        """Return current queue size."""
        with self.lock:
            return len(self.queue)

    def stats(self) -> dict:
        """Return queue statistics."""
        with self.lock:
            completed = sum(1 for r in self.request_index.values() if r.status == RequestStatus.COMPLETED)
            failed = sum(1 for r in self.request_index.values() if r.status == RequestStatus.FAILED)
            queued = sum(1 for r in self.request_index.values() if r.status == RequestStatus.QUEUED)
            processing = sum(1 for r in self.request_index.values() if r.status == RequestStatus.PROCESSING)

            return {
                "queue_size": len(self.queue),
                "queued": queued,
                "processing": processing,
                "completed": completed,
                "failed": failed,
                "total_requests": len(self.request_index),
            }

    def clear(self) -> None:
        """Clear queue and history."""
        with self.lock:
            self.queue.clear()
            self.request_index.clear()
            logger.info("Cleared request queue")
