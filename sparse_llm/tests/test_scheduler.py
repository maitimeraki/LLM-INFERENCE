"""Tests for request scheduler, KV cache paging, and continuous batching."""

import pytest
import torch

from sparse_llm.inference.scheduler import (
    KVCacheState,
    KVPageAllocator,
    RequestScheduler,
    RequestSchedulerConfig,
    RequestState,
    SequenceState,
    KVPageConfig,
)
from sparse_llm.inference.continuous_batch import (
    ContinuousBatchScheduler,
    ContinuousGenerationEngine,
    BatchScheduleConfig,
    DecodeBatch,
)


class TestKVCacheState:
    """Tests for KV cache state tracking."""

    def test_init(self) -> None:
        kv = KVCacheState(request_id=1)
        assert kv.request_id == 1
        assert kv.allocated_pages == []
        assert kv.page_occupancy == 0

    def test_num_pages_needed(self) -> None:
        kv = KVCacheState(request_id=1)
        assert kv.num_pages_needed(0, 128) == 0
        assert kv.num_pages_needed(128, 128) == 1
        assert kv.num_pages_needed(129, 128) == 2
        assert kv.num_pages_needed(256, 128) == 2

    def test_can_allocate(self) -> None:
        kv = KVCacheState(request_id=1, page_occupancy=64)
        assert kv.can_allocate(64, 128, 2)  # 64+64=128, needs 1 page
        assert not kv.can_allocate(256, 128, 2)  # 64+256=320, needs 3 pages but max is 2

    def test_can_allocate_boundary(self) -> None:
        kv = KVCacheState(request_id=1, page_occupancy=128)
        assert kv.can_allocate(128, 128, 2)
        assert not kv.can_allocate(129, 128, 2)


class TestSequenceState:
    """Tests for per-request sequence state."""

    def test_init(self) -> None:
        prompt = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        seq = SequenceState(request_id=1, prompt_tokens=prompt)
        assert seq.request_id == 1
        assert seq.generated_tokens == []
        assert seq.state == RequestState.ADMITTED

    def test_total_tokens_generated(self) -> None:
        prompt = torch.tensor([1, 2, 3], dtype=torch.long)
        seq = SequenceState(request_id=1, prompt_tokens=prompt)
        assert seq.total_tokens_generated() == 0
        seq.generated_tokens = [10, 20, 30]
        assert seq.total_tokens_generated() == 3

    def test_input_length(self) -> None:
        prompt = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        seq = SequenceState(request_id=1, prompt_tokens=prompt)
        assert seq.input_length() == 5

    def test_total_sequence_length(self) -> None:
        prompt = torch.tensor([1, 2, 3], dtype=torch.long)
        seq = SequenceState(request_id=1, prompt_tokens=prompt)
        assert seq.total_sequence_length() == 3
        seq.generated_tokens = [10, 20]
        assert seq.total_sequence_length() == 5

    def test_should_stop_max_tokens(self) -> None:
        prompt = torch.tensor([1, 2], dtype=torch.long)
        seq = SequenceState(
            request_id=1,
            prompt_tokens=prompt,
            stopping_criteria={"max_tokens": 5},
        )
        seq.generated_tokens = [10, 20, 30]
        assert not seq.should_stop()  # 3 < 5
        seq.generated_tokens.append(40)
        seq.generated_tokens.append(50)
        assert seq.should_stop()  # 5 >= 5


class TestKVPageAllocator:
    """Tests for KV page allocation."""

    def test_init(self) -> None:
        alloc = KVPageAllocator(total_pages=100)
        assert alloc.total_pages == 100
        assert alloc.free_pages_count() == 100

    def test_allocate(self) -> None:
        alloc = KVPageAllocator(total_pages=100)
        pages = alloc.allocate(10)
        assert pages is not None
        assert len(pages) == 10
        assert alloc.free_pages_count() == 90

    def test_allocate_all(self) -> None:
        alloc = KVPageAllocator(total_pages=10)
        pages = alloc.allocate(10)
        assert pages is not None
        assert len(pages) == 10
        assert alloc.free_pages_count() == 0

    def test_allocate_exceeds_capacity(self) -> None:
        alloc = KVPageAllocator(total_pages=10)
        pages = alloc.allocate(11)
        assert pages is None
        assert alloc.free_pages_count() == 10

    def test_release(self) -> None:
        alloc = KVPageAllocator(total_pages=100)
        pages = alloc.allocate(10)
        assert alloc.free_pages_count() == 90
        alloc.release(pages)
        assert alloc.free_pages_count() == 100

    def test_utilization(self) -> None:
        alloc = KVPageAllocator(total_pages=100)
        assert alloc.utilization() == 0.0
        alloc.allocate(50)
        assert alloc.utilization() == 0.5
        alloc.allocate(50)
        assert alloc.utilization() == 1.0


class TestRequestScheduler:
    """Tests for request scheduler."""

    def test_init(self) -> None:
        sched = RequestScheduler()
        assert len(sched.get_active_requests()) == 0
        assert sched._next_request_id == 0

    def test_admit_request(self) -> None:
        sched = RequestScheduler()
        prompt = torch.tensor([1, 2, 3], dtype=torch.long)
        req_id = sched.admit_request(prompt)
        assert req_id == 0
        assert req_id in sched.get_active_requests()

    def test_admit_multiple_requests(self) -> None:
        sched = RequestScheduler()
        prompt = torch.tensor([1, 2, 3], dtype=torch.long)
        ids = [sched.admit_request(prompt) for _ in range(5)]
        assert len(ids) == 5
        assert all(id in sched.get_active_requests() for id in ids)

    def test_admit_exceeds_capacity(self) -> None:
        config = RequestSchedulerConfig(max_requests=3)
        sched = RequestScheduler(config)
        prompt = torch.tensor([1, 2, 3], dtype=torch.long)

        for _ in range(3):
            req_id = sched.admit_request(prompt)
            assert req_id is not None

        # Fourth should fail
        req_id = sched.admit_request(prompt)
        assert req_id is None

    def test_get_sequence(self) -> None:
        sched = RequestScheduler()
        prompt = torch.tensor([1, 2, 3], dtype=torch.long)
        req_id = sched.admit_request(prompt)
        seq = sched.get_sequence(req_id)
        assert seq is not None
        assert seq.request_id == req_id

    def test_update_sequence_state(self) -> None:
        sched = RequestScheduler()
        prompt = torch.tensor([1, 2, 3], dtype=torch.long)
        req_id = sched.admit_request(prompt)
        assert sched.update_sequence(req_id, RequestState.DECODE)
        seq = sched.get_sequence(req_id)
        assert seq.state == RequestState.DECODE

    def test_add_generated_token(self) -> None:
        sched = RequestScheduler()
        prompt = torch.tensor([1, 2, 3], dtype=torch.long)
        req_id = sched.admit_request(prompt)
        assert sched.add_generated_token(req_id, 100)
        seq = sched.get_sequence(req_id)
        assert seq.generated_tokens == [100]

    def test_mark_prefill_done(self) -> None:
        sched = RequestScheduler()
        prompt = torch.tensor([1, 2, 3], dtype=torch.long)
        req_id = sched.admit_request(prompt)
        assert sched.mark_prefill_done(req_id)
        seq = sched.get_sequence(req_id)
        assert seq.prefill_done
        assert seq.state == RequestState.DECODE

    def test_complete_request(self) -> None:
        sched = RequestScheduler()
        prompt = torch.tensor([1, 2, 3], dtype=torch.long)
        req_id = sched.admit_request(prompt)
        assert sched.complete_request(req_id)
        assert req_id not in sched.get_active_requests()

    def test_stats(self) -> None:
        sched = RequestScheduler()
        prompt = torch.tensor([1, 2, 3], dtype=torch.long)
        sched.admit_request(prompt)
        stats = sched.stats()
        assert stats["total_admitted"] == 1
        assert stats["active_requests"] == 1


class TestDecodeBatch:
    """Tests for decode batch."""

    def test_init(self) -> None:
        batch = DecodeBatch(batch_id=0)
        assert batch.batch_id == 0
        assert batch.request_ids == []
        assert batch.total_tokens() == 0

    def test_total_tokens(self) -> None:
        batch = DecodeBatch(batch_id=0)
        batch.request_ids = [1, 2, 3]
        batch.token_counts = {1: 5, 2: 10, 3: 3}
        assert batch.total_tokens() == 18

    def test_average_length(self) -> None:
        batch = DecodeBatch(batch_id=0)
        batch.request_ids = [1, 2, 3]
        batch.token_counts = {1: 10, 2: 20, 3: 30}
        assert batch.average_length() == 20.0


class TestContinuousBatchScheduler:
    """Tests for continuous batch scheduler."""

    def test_init(self) -> None:
        sched = RequestScheduler()
        batch_sched = ContinuousBatchScheduler(sched)
        assert batch_sched._next_batch_id == 0

    def test_schedule_empty(self) -> None:
        sched = RequestScheduler()
        batch_sched = ContinuousBatchScheduler(sched)
        batch = batch_sched.schedule_batch()
        assert batch is None

    def test_schedule_decode_requests(self) -> None:
        sched = RequestScheduler()
        batch_sched = ContinuousBatchScheduler(sched)

        prompt = torch.tensor([1, 2, 3], dtype=torch.long)
        req_id = sched.admit_request(prompt)
        sched.mark_prefill_done(req_id)

        batch = batch_sched.schedule_batch()
        assert batch is not None
        assert req_id in batch.request_ids

    def test_execute_decode_step(self) -> None:
        sched = RequestScheduler()
        batch_sched = ContinuousBatchScheduler(sched)

        prompt = torch.tensor([1, 2, 3], dtype=torch.long)
        req_id = sched.admit_request(prompt)
        sched.mark_prefill_done(req_id)

        batch = batch_sched.schedule_batch()
        result = batch_sched.execute_decode_step(batch)
        assert result is not None
        assert req_id in result

    def test_batch_metrics(self) -> None:
        sched = RequestScheduler()
        batch_sched = ContinuousBatchScheduler(sched)

        prompt = torch.tensor([1, 2, 3], dtype=torch.long)
        req_id = sched.admit_request(prompt)
        sched.mark_prefill_done(req_id)

        batch = batch_sched.schedule_batch()
        batch_sched.execute_decode_step(batch)

        metrics = batch_sched.get_batch_metrics()
        assert len(metrics) == 1
        assert metrics[0].num_requests == 1


class TestContinuousGenerationEngine:
    """Tests for continuous generation engine."""

    def test_init(self) -> None:
        sched = RequestScheduler()
        engine = ContinuousGenerationEngine(sched)
        assert engine._total_batches_processed == 0

    def test_generate_batch_step(self) -> None:
        sched = RequestScheduler()
        engine = ContinuousGenerationEngine(sched)

        prompt = torch.tensor([1, 2, 3], dtype=torch.long)
        req_id = sched.admit_request(prompt)
        sched.mark_prefill_done(req_id)

        result = engine.generate_batch_step()
        assert result is not None
        assert req_id in result

    def test_stats(self) -> None:
        sched = RequestScheduler()
        engine = ContinuousGenerationEngine(sched)

        prompt = torch.tensor([1, 2, 3], dtype=torch.long)
        sched.admit_request(prompt)

        stats = engine.stats()
        assert "scheduler" in stats
        assert "batch_scheduler" in stats
        assert "total_batches_processed" in stats


# Integration tests
class TestSchedulerIntegration:
    """Integration tests for scheduler and batch components."""

    def test_full_generation_cycle(self) -> None:
        """Test complete request lifecycle: admit → prefill → decode → complete."""
        sched = RequestScheduler()
        batch_sched = ContinuousBatchScheduler(sched)

        prompt = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        req_id = sched.admit_request(
            prompt, stopping_criteria={"max_tokens": 5}
        )

        assert req_id is not None
        assert req_id in sched.get_active_requests()

        # Transition to decode
        assert sched.mark_prefill_done(req_id)

        # Generate tokens until stopping
        for _ in range(10):
            batch = batch_sched.schedule_batch()
            if batch is None:
                break
            result = batch_sched.execute_decode_step(batch)
            if result is None:
                break

        # Should be complete
        assert req_id not in sched.get_active_requests()

    def test_multi_request_batching(self) -> None:
        """Test multiple requests batched together."""
        sched = RequestScheduler()
        batch_sched = ContinuousBatchScheduler(sched)

        # Admit 3 requests
        req_ids = []
        for i in range(3):
            prompt = torch.tensor([1, 2, 3], dtype=torch.long)
            req_id = sched.admit_request(prompt)
            sched.mark_prefill_done(req_id)
            req_ids.append(req_id)

        # Schedule batch
        batch = batch_sched.schedule_batch()
        assert batch is not None
        assert len(batch.request_ids) == 3

        # Execute decode
        result = batch_sched.execute_decode_step(batch)
        assert len(result) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
