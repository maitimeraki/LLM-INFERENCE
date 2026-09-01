"""Safe, local checkpoint indexing for architecture-validated expert tensors."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

import torch
from safetensors import safe_open

from sparse_llm.cache.expert_cache import ExpertKey


@dataclass(frozen=True)
class ExpertTensorMapping:
    """Architecture-specific mapping from a tensor name to an expert key."""

    key: ExpertKey
    tensor_name: str


@dataclass(frozen=True)
class TensorLocation:
    """Location and shape metadata for one safe checkpoint tensor.

    ``data_offsets`` are the offsets recorded by the safetensors header,
    relative to the start of the file payload. No tensor bytes are read while
    constructing an index.
    """

    name: str
    file: Path
    shape: tuple[int, ...]
    dtype: str
    byte_size: int
    data_offsets: tuple[int, int] = (0, 0)
    checksum: str | None = None

    @property
    def offsets(self) -> tuple[int, int]:
        """Compatibility alias for callers that use the shorter name."""
        return self.data_offsets

    @property
    def file_offset(self) -> tuple[int, int]:
        """Return absolute byte offsets including the safetensors header."""
        return self.data_offsets[0], self.data_offsets[1]


@dataclass(frozen=True)
class ExpertRecord:
    """All tensors required to materialize one layer-aware expert."""

    key: ExpertKey
    tensors: tuple[TensorLocation, ...]

    @property
    def byte_size(self) -> int:
        return sum(tensor.byte_size for tensor in self.tensors)


_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
    "C64": 8,
    "C128": 16,
}

_TORCH_DTYPE_NAMES = {
    "BOOL": "BOOL",
    "U8": "UINT8",
    "I8": "INT8",
    "I16": "INT16",
    "I32": "INT32",
    "I64": "INT64",
    "F8_E4M3": "FLOAT8_E4M3FN",
    "F8_E5M2": "FLOAT8_E5M2",
    "F16": "FLOAT16",
    "BF16": "BFLOAT16",
    "F32": "FLOAT32",
    "F64": "FLOAT64",
}


class CheckpointIndex:
    """Index safe local tensors without constructing a full model."""

    def __init__(
        self,
        directory: str | Path,
        tensors: Mapping[str, TensorLocation],
        experts: Mapping[ExpertKey, Iterable[str]],
    ) -> None:
        self.directory = Path(directory)
        self._tensors = dict(tensors)
        grouped: dict[ExpertKey, tuple[TensorLocation, ...]] = {}
        mapped_names: set[str] = set()
        for key, names in experts.items():
            locations = []
            for name in names:
                if name in mapped_names:
                    raise ValueError(f"tensor {name!r} was mapped more than once")
                if name not in self._tensors:
                    raise ValueError(f"expert mapping references missing tensor {name!r}")
                mapped_names.add(name)
                locations.append(self._tensors[name])
            if not locations:
                raise ValueError(f"expert {key!r} has no tensors")
            grouped[key] = tuple(locations)
        self._experts = {
            key: ExpertRecord(key, locations) for key, locations in grouped.items()
        }
        self._shared_names = tuple(sorted(set(self._tensors) - mapped_names))

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        mapper: Callable[[str], ExpertTensorMapping | None],
        *,
        expected_keys: Iterable[ExpertKey] | None = None,
        expected_num_layers: int | None = None,
        expected_num_experts: int | None = None,
    ) -> "CheckpointIndex":
        """Build an index using an architecture-specific tensor-name mapper."""
        if not callable(mapper):
            raise TypeError("mapper must be callable")
        root = Path(directory)
        if not root.is_dir():
            raise FileNotFoundError(f"checkpoint directory does not exist: {root}")
        files = cls._checkpoint_files(root)
        if not files:
            unsafe = sorted(
                path
                for pattern in ("*.bin", "*.pt", "*.pth", "*.ckpt")
                for path in root.glob(pattern)
                if path.is_file()
            )
            if unsafe:
                raise ValueError(
                    "unsafe checkpoint format found; safetensors is required by default: "
                    + ", ".join(path.name for path in unsafe)
                )
            raise FileNotFoundError(f"no safetensors checkpoint found in {root}")

        locations: dict[str, TensorLocation] = {}
        for path in files:
            header = cls._read_header(path)
            try:
                handle = safe_open(str(path), framework="pt", device="cpu")
                names = list(handle.keys())
                for name in names:
                    if name in locations:
                        raise ValueError(f"duplicate checkpoint tensor {name!r}")
                    metadata = header.get(name)
                    if not isinstance(metadata, dict):
                        raise ValueError(f"tensor {name!r} is missing a valid safetensors header")
                    shape = tuple(int(dimension) for dimension in metadata.get("shape", ()))
                    dtype = str(metadata.get("dtype", "")).upper()
                    offsets = metadata.get("data_offsets")
                    if dtype not in _DTYPE_BYTES or not isinstance(offsets, list) or len(offsets) != 2:
                        raise ValueError(f"invalid safetensors metadata for tensor {name!r}")
                    start, end = offsets
                    if (
                        any(not isinstance(value, int) or isinstance(value, bool) for value in offsets)
                        or start < 0
                        or end < start
                        or end - start != _product(shape) * _DTYPE_BYTES[dtype]
                    ):
                        raise ValueError(f"invalid byte offsets for tensor {name!r}")
                    header_checksum = header.get("__metadata__", {}).get(f"sha256:{name}")
                    locations[name] = TensorLocation(
                        name=name,
                        file=path,
                        shape=shape,
                        dtype=dtype,
                        byte_size=end - start,
                        data_offsets=(start, end),
                        checksum=header_checksum if isinstance(header_checksum, str) else None,
                    )
            except ValueError:
                raise
            except Exception as error:
                raise ValueError(f"cannot inspect safe checkpoint {path}: {error}") from error

        grouped: dict[ExpertKey, list[str]] = {}
        mapped_names: set[str] = set()
        for name in sorted(locations):
            mapping = mapper(name)
            if mapping is None:
                continue
            key = cls._validate_key(mapping.key)
            if mapping.tensor_name != name:
                raise ValueError(
                    f"expert mapper returned tensor name {mapping.tensor_name!r} for {name!r}"
                )
            if name in mapped_names:
                raise ValueError(f"tensor {name!r} was mapped more than once")
            mapped_names.add(name)
            grouped.setdefault(key, []).append(name)
        if expected_keys is not None:
            expected = {cls._validate_key(key) for key in expected_keys}
            missing = sorted(expected - set(grouped))
            unexpected = sorted(set(grouped) - expected)
            if missing:
                raise ValueError(f"checkpoint is missing expected experts: {missing}")
            if unexpected:
                raise ValueError(f"checkpoint contains unexpected experts: {unexpected}")
        if expected_num_layers is not None or expected_num_experts is not None:
            cls._validate_dimensions(
                grouped,
                expected_num_layers=expected_num_layers,
                expected_num_experts=expected_num_experts,
            )
        if not grouped:
            raise ValueError("architecture mapper found no expert tensors")
        return cls(root, locations, grouped)

    @staticmethod
    def _read_header(path: Path) -> dict:
        """Read and validate only the safetensors JSON header."""
        try:
            with path.open("rb") as stream:
                raw_length = stream.read(8)
                if len(raw_length) != 8:
                    raise ValueError("file has no complete safetensors header length")
                header_length = struct.unpack("<Q", raw_length)[0]
                if header_length > 64 * 1024 * 1024:
                    raise ValueError("safetensors header exceeds the 64 MiB safety limit")
                raw_header = stream.read(header_length)
                if len(raw_header) != header_length:
                    raise ValueError("file has a truncated safetensors header")
            header = json.loads(raw_header.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, struct.error) as error:
            raise ValueError(f"invalid safetensors header in {path}: {error}") from error
        if not isinstance(header, dict):
            raise ValueError(f"safetensors header in {path} must be an object")
        return header

    @staticmethod
    def _checkpoint_files(root: Path) -> list[Path]:
        index_path = root / "model.safetensors.index.json"
        if index_path.exists():
            try:
                payload = json.loads(index_path.read_text(encoding="utf-8"))
                weight_map = payload["weight_map"]
                if not isinstance(weight_map, dict):
                    raise ValueError("weight_map must be an object")
                root_resolved = root.resolve()
                files = []
                for name in sorted(set(weight_map.values())):
                    if not isinstance(name, str):
                        raise ValueError("weight_map file names must be strings")
                    candidate = (root / name).resolve()
                    try:
                        candidate.relative_to(root_resolved)
                    except ValueError as error:
                        raise ValueError(f"checkpoint shard escapes directory: {name!r}") from error
                    files.append(candidate)
                missing = [path for path in files if not path.is_file()]
                if missing:
                    raise FileNotFoundError(
                        "checkpoint index references missing safetensors: "
                        + ", ".join(path.name for path in missing)
                    )
                if any(path.suffix != ".safetensors" for path in files):
                    raise ValueError("checkpoint index references a non-safetensors shard")
                return files
            except FileNotFoundError:
                raise
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid safetensors index {index_path}: {error}") from error
        return sorted(path for path in root.glob("*.safetensors") if path.is_file())

    @staticmethod
    def _validate_key(key: object) -> ExpertKey:
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError("expert mapping key must be ExpertKey (layer_index, expert_index)")
        layer, expert = key
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in key):
            raise ValueError("expert mapping key indices must be non-negative integers")
        return layer, expert

    @classmethod
    def _validate_dimensions(
        cls,
        grouped: Mapping[ExpertKey, Iterable[str]],
        *,
        expected_num_layers: int | None,
        expected_num_experts: int | None,
    ) -> None:
        for key in grouped:
            layer, expert = cls._validate_key(key)
            if expected_num_layers is not None and layer >= expected_num_layers:
                raise ValueError(f"expert key {key!r} exceeds configured layer count")
            if expected_num_experts is not None and expert >= expected_num_experts:
                raise ValueError(f"expert key {key!r} exceeds configured expert count")

    def validate_config(
        self,
        config: object,
        *,
        layer_names: tuple[str, ...] = ("num_hidden_layers", "num_layers", "n_layer"),
        expert_names: tuple[str, ...] = (
            "num_local_experts",
            "num_experts",
            "n_routed_experts",
        ),
    ) -> None:
        """Validate discovered key bounds against a loaded architecture config."""
        layer_count = _first_positive_int(config, layer_names)
        expert_count = _first_positive_int(config, expert_names)
        self._validate_dimensions(
            self._experts,
            expected_num_layers=layer_count,
            expected_num_experts=expert_count,
        )
        if layer_count is not None and {key[0] for key in self._experts} != set(range(layer_count)):
            raise ValueError("checkpoint expert index does not cover every configured layer")

    @property
    def expert_keys(self) -> tuple[ExpertKey, ...]:
        return tuple(sorted(self._experts))

    @property
    def shared_tensor_names(self) -> tuple[str, ...]:
        return self._shared_names

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tensors))

    def expert(self, key: ExpertKey) -> ExpertRecord:
        key = self._validate_key(key)
        try:
            return self._experts[key]
        except KeyError as error:
            raise KeyError(f"expert {key!r} is not present in checkpoint index") from error

    def tensor(self, name: str) -> TensorLocation:
        try:
            return self._tensors[name]
        except KeyError as error:
            raise KeyError(f"tensor {name!r} is not present in checkpoint index") from error

    def load_expert(self, key: ExpertKey) -> dict[str, torch.Tensor]:
        """Read only the indexed tensors belonging to one expert."""
        record = self.expert(key)
        loaded: dict[str, torch.Tensor] = {}
        handles: dict[Path, object] = {}
        try:
            for location in record.tensors:
                handle = handles.get(location.file)
                if handle is None:
                    handle = safe_open(str(location.file), framework="pt", device="cpu")
                    handles[location.file] = handle
                tensor = handle.get_tensor(location.name)
                actual_shape = tuple(tensor.shape)
                actual_dtype = str(tensor.dtype).split(".")[-1].upper()
                actual_bytes = tensor.numel() * tensor.element_size()
                expected_dtype = _TORCH_DTYPE_NAMES.get(location.dtype, location.dtype)
                if actual_shape != location.shape or actual_dtype != expected_dtype:
                    raise ValueError(
                        f"tensor {location.name!r} metadata changed: expected "
                        f"shape={location.shape}, dtype={location.dtype}; got "
                        f"shape={actual_shape}, dtype={actual_dtype}"
                    )
                if actual_bytes != location.byte_size:
                    raise ValueError(
                        f"tensor {location.name!r} byte size changed: expected "
                        f"{location.byte_size}, got {actual_bytes}"
                    )
                loaded[location.name] = tensor.detach().cpu()
        finally:
            for handle in handles.values():
                close = getattr(handle, "__exit__", None)
                if callable(close):
                    close(None, None, None)
        return loaded


def _first_positive_int(source: object, names: Iterable[str]) -> int | None:
    for name in names:
        value = getattr(source, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _product(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        if dimension < 0:
            raise ValueError("tensor shapes cannot contain negative dimensions")
        result *= dimension
    return result


__all__ = [
    "CheckpointIndex",
    "ExpertRecord",
    "ExpertTensorMapping",
    "TensorLocation",
]
