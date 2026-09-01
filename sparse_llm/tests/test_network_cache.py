"""Tests for network-backed expert cache."""

import hashlib
import tempfile
from pathlib import Path

import pytest

from sparse_llm.storage.network_cache import (
    ExpertPageIdentity,
    ExpertPageRecord,
    NetworkCache,
)


class TestExpertPageIdentity:
    """Test ExpertPageIdentity dataclass."""

    def test_creation(self):
        identity = ExpertPageIdentity(
            checkpoint_version="v1.0",
            layer_id=5,
            expert_id=10,
            quantization_format="int8",
        )
        assert identity.checkpoint_version == "v1.0"
        assert identity.layer_id == 5
        assert identity.expert_id == 10
        assert identity.quantization_format == "int8"

    def test_unique_id_generation(self):
        identity = ExpertPageIdentity(
            checkpoint_version="v1.0", layer_id=0, expert_id=0
        )
        uid = identity.unique_id
        assert "v1.0" in uid
        assert "layer_0000" in uid
        assert "expert_0000" in uid

    def test_unique_id_distinct(self):
        id1 = ExpertPageIdentity(
            checkpoint_version="v1.0", layer_id=0, expert_id=0
        ).unique_id
        id2 = ExpertPageIdentity(
            checkpoint_version="v1.0", layer_id=1, expert_id=0
        ).unique_id
        assert id1 != id2

    def test_frozen(self):
        identity = ExpertPageIdentity(
            checkpoint_version="v1.0", layer_id=5, expert_id=10
        )
        with pytest.raises(AttributeError):
            identity.layer_id = 6


class TestExpertPageRecord:
    """Test ExpertPageRecord."""

    def test_creation(self):
        identity = ExpertPageIdentity(
            checkpoint_version="v1.0", layer_id=0, expert_id=0
        )
        record = ExpertPageRecord(identity=identity, byte_size=1024)
        assert record.identity == identity
        assert record.byte_size == 1024
        assert record.checksum_sha256 is None

    def test_creation_with_checksum(self):
        identity = ExpertPageIdentity(
            checkpoint_version="v1.0", layer_id=0, expert_id=0
        )
        checksum = "abc123"
        record = ExpertPageRecord(
            identity=identity, byte_size=1024, checksum_sha256=checksum
        )
        assert record.checksum_sha256 == checksum

    def test_validate_checksum_none(self):
        identity = ExpertPageIdentity(
            checkpoint_version="v1.0", layer_id=0, expert_id=0
        )
        record = ExpertPageRecord(identity=identity, byte_size=10)
        assert record.validate_checksum(b"anything") is True

    def test_validate_checksum_matches(self):
        data = b"test data"
        checksum = hashlib.sha256(data).hexdigest()
        identity = ExpertPageIdentity(
            checkpoint_version="v1.0", layer_id=0, expert_id=0
        )
        record = ExpertPageRecord(
            identity=identity, byte_size=len(data), checksum_sha256=checksum
        )
        assert record.validate_checksum(data) is True

    def test_validate_checksum_fails(self):
        identity = ExpertPageIdentity(
            checkpoint_version="v1.0", layer_id=0, expert_id=0
        )
        checksum = "wronghash"
        record = ExpertPageRecord(
            identity=identity, byte_size=10, checksum_sha256=checksum
        )
        assert record.validate_checksum(b"test data") is False

    def test_compute_checksum(self):
        data = b"test data"
        identity = ExpertPageIdentity(
            checkpoint_version="v1.0", layer_id=0, expert_id=0
        )
        record = ExpertPageRecord(identity=identity, byte_size=len(data))
        computed = record.compute_checksum(data)
        expected = hashlib.sha256(data).hexdigest()
        assert computed == expected


class TestNetworkCache:
    """Test NetworkCache."""

    def test_initialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = NetworkCache(local_cache_dir=tmpdir)
            assert Path(tmpdir).exists()
            assert cache.enable_checksum is True

    def test_register_page(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = NetworkCache(local_cache_dir=tmpdir)
            identity = ExpertPageIdentity(
                checkpoint_version="v1.0", layer_id=0, expert_id=0
            )
            cache.register_page(identity, byte_size=1024, checksum="abc123")
            assert identity.unique_id in cache.page_records

    def test_get_page_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = NetworkCache(local_cache_dir=tmpdir)
            identity = ExpertPageIdentity(
                checkpoint_version="v1.0", layer_id=0, expert_id=0
            )
            path = cache.get_page_path(identity)
            assert str(path).endswith(".pt")
            assert tmpdir in str(path)

    def test_save_and_load_expert(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = NetworkCache(local_cache_dir=tmpdir)
            identity = ExpertPageIdentity(
                checkpoint_version="v1.0", layer_id=0, expert_id=0
            )
            data = b"test expert data"
            cache.save_expert(identity, data)

            loaded = cache.load_expert(identity)
            assert loaded == data

    def test_load_expert_from_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = NetworkCache(local_cache_dir=tmpdir)
            identity = ExpertPageIdentity(
                checkpoint_version="v1.0", layer_id=0, expert_id=0
            )
            data = b"test data"
            cache.save_expert(identity, data)

            loaded = cache.load_expert(identity)
            assert cache.cache_stats["hits"] == 1
            assert cache.cache_stats["misses"] == 0

    def test_load_expert_checksum_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = NetworkCache(local_cache_dir=tmpdir, enable_checksum=True)
            identity = ExpertPageIdentity(
                checkpoint_version="v1.0", layer_id=0, expert_id=0
            )
            data = b"original data"
            cache.save_expert(identity, data)

            loaded = cache.load_expert(identity)
            assert loaded == data
            assert cache.cache_stats["checksum_failures"] == 0

    def test_load_expert_checksum_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = NetworkCache(local_cache_dir=tmpdir, enable_checksum=True)
            identity = ExpertPageIdentity(
                checkpoint_version="v1.0", layer_id=0, expert_id=0
            )
            data = b"original"
            cache.register_page(
                identity, byte_size=len(data), checksum="wrongchecksum"
            )
            path = cache.get_page_path(identity)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

            with pytest.raises(ValueError, match="Checksum mismatch"):
                cache.load_expert(identity)
            assert cache.cache_stats["checksum_failures"] == 1

    def test_load_expert_remote_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            remote_data = b"remote expert data"

            def mock_remote_fetch(identity):
                return remote_data

            cache = NetworkCache(
                local_cache_dir=tmpdir, remote_fetch_fn=mock_remote_fetch
            )
            identity = ExpertPageIdentity(
                checkpoint_version="v1.0", layer_id=0, expert_id=0
            )
            cache.register_page(identity, byte_size=len(remote_data))

            loaded = cache.load_expert(identity)
            assert loaded == remote_data
            assert cache.cache_stats["misses"] == 1
            assert cache.cache_stats["remote_fetches"] == 1

    def test_load_expert_no_remote_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = NetworkCache(local_cache_dir=tmpdir, remote_fetch_fn=None)
            identity = ExpertPageIdentity(
                checkpoint_version="v1.0", layer_id=0, expert_id=0
            )
            cache.register_page(identity, byte_size=1024)

            with pytest.raises(RuntimeError, match="remote fetch unavailable"):
                cache.load_expert(identity)

    def test_load_expert_not_registered(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = NetworkCache(local_cache_dir=tmpdir)
            identity = ExpertPageIdentity(
                checkpoint_version="v1.0", layer_id=0, expert_id=0
            )
            with pytest.raises(KeyError):
                cache.load_expert(identity)

    def test_cache_statistics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = NetworkCache(local_cache_dir=tmpdir)
            identity = ExpertPageIdentity(
                checkpoint_version="v1.0", layer_id=0, expert_id=0
            )
            cache.register_page(identity, byte_size=100)
            data = b"x" * 100
            cache.save_expert(identity, data)
            cache.load_expert(identity)

            stats = cache.get_stats()
            assert stats["registered_pages"] == 1
            assert stats["local_cache_dir"] == tmpdir

    def test_clear_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = NetworkCache(local_cache_dir=tmpdir)
            cache.cache_stats["hits"] = 10
            cache.clear_stats()
            assert cache.cache_stats["hits"] == 0
            assert cache.cache_stats["misses"] == 0

    def test_remote_fetch_checksum_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            remote_data = b"remote data"

            def mock_remote_fetch(identity):
                return remote_data

            cache = NetworkCache(
                local_cache_dir=tmpdir, remote_fetch_fn=mock_remote_fetch
            )
            identity = ExpertPageIdentity(
                checkpoint_version="v1.0", layer_id=0, expert_id=0
            )
            checksum = hashlib.sha256(remote_data).hexdigest()
            cache.register_page(identity, byte_size=len(remote_data), checksum=checksum)

            loaded = cache.load_expert(identity)
            assert loaded == remote_data

    def test_checksum_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = NetworkCache(local_cache_dir=tmpdir, enable_checksum=False)
            identity = ExpertPageIdentity(
                checkpoint_version="v1.0", layer_id=0, expert_id=0
            )
            cache.register_page(
                identity, byte_size=10, checksum="wrongchecksum"
            )
            path = cache.get_page_path(identity)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"data")

            # Should not raise despite checksum mismatch
            loaded = cache.load_expert(identity)
            assert loaded == b"data"
