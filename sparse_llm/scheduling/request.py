import logging
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import uuid
import time

logger = logging.getLogger(__name__)


class RequestStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class InferenceRequest:
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    user_id: Optional[str] = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    status: RequestStatus = RequestStatus.QUEUED
    result: Optional[str] = None
    error: Optional[str] = None
    processing_start_time: Optional[float] = None
    processing_end_time: Optional[float] = None

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "status": self.status.value,
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "processing_start_time": self.processing_start_time,
            "processing_end_time": self.processing_end_time,
        }
