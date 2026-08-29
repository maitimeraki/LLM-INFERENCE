#!/usr/bin/env python3
"""SparseLLM v2: Multi-user sparse MoE inference system."""

import logging
import time
import torch

from sparse_llm.models import SparseMoELayer
from sparse_llm.storage import LocalSSDStorage
from sparse_llm.inference import InferenceEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def demo_basic_inference():
    """Demo: Basic multi-user inference with sparse MoE."""
    logger.info("=" * 70)
    logger.info("DEMO: Multi-User Sparse MoE Inference")
    logger.info("=" * 70)

    model = SparseMoELayer(
        input_dim=768,
        num_experts=22,
        expert_dim=3072,
        top_k=4
    )
    logger.info(f"Initialized SparseMoELayer: {model.num_experts} experts, top-k={model.top_k}")

    storage = LocalSSDStorage(base_path="./expert_storage")
    logger.info("Initialized LocalSSDStorage")

    engine = InferenceEngine(
        model=model,
        storage=storage,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_queue_size=1000,
        max_cached_experts=22
    )
    logger.info(f"Initialized InferenceEngine on {engine.device}")

    engine.start()
    logger.info("Started inference engine worker thread")

    user_prompts = [
        ("user_1", "The quick brown fox jumps over the lazy dog", 50),
        ("user_2", "SparseLLM enables efficient inference for MoE models", 50),
        ("user_3", "Loading only active experts reduces memory footprint", 50),
    ]

    request_ids = []
    for user_id, prompt, max_tokens in user_prompts:
        req_id = engine.submit_request(user_id, prompt, max_tokens)
        request_ids.append(req_id)
        logger.info(f"Submitted request {req_id} from {user_id}")

    logger.info("\nMonitoring request status...")
    for _ in range(5):
        time.sleep(0.5)
        stats = engine.stats()
        logger.info(f"Queue: {stats['queue']}")
        logger.info(f"Cache: {stats['cache']}")

    logger.info("\nFinal request status:")
    for req_id in request_ids:
        status = engine.get_request_status(req_id)
        logger.info(f"Request {req_id}: {status}")

    engine.stop()
    logger.info("Stopped inference engine")

    return True


def demo_cache_efficiency():
    """Demo: Expert cache with LRU + scoring."""
    logger.info("=" * 70)
    logger.info("DEMO: Expert Cache Efficiency")
    logger.info("=" * 70)

    model = SparseMoELayer(
        input_dim=768,
        num_experts=22,
        expert_dim=3072,
        top_k=4
    )

    storage = LocalSSDStorage(base_path="./expert_storage")
    engine = InferenceEngine(
        model=model,
        storage=storage,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_cached_experts=8
    )

    engine.start()

    logger.info("Submitting batch requests to trigger cache behavior...")
    for i in range(5):
        req_id = engine.submit_request(f"user_{i}", f"Prompt {i}", 20)
        logger.info(f"Submitted request {req_id}")
        time.sleep(0.1)

    time.sleep(1.0)

    stats = engine.stats()
    logger.info(f"Cache stats: {stats['cache']}")
    logger.info(f"Prefetch stats: {stats['prefetch']}")

    engine.stop()
    return True


def main():
    """Run all demos."""
    logger.info("SparseLLM v2: Multi-User Sparse MoE Inference System\n")

    try:
        demo_basic_inference()
        print()
        demo_cache_efficiency()
        print()
        logger.info("[OK] All demos completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
