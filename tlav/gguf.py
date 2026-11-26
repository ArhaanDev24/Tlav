"""Pure-Python GGUF header reader.

Reads only the metadata + tensor table at the start of a .gguf file (local or
remote via HTTP Range requests, a few MB at most), which is enough to know a
model's exact architecture and parameter count without downloading it.
"""

import os
import struct
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

GGUF_MAGIC = b"GGUF"

# GGUF metadata value types
_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9


class _Source:
    """Byte source that reads lazily from a file or URL."""

    def __init__(self, path_or_url: str, chunk: int = 4 << 20, max_bytes: int = 64 << 20):
        self.loc = path_or_url
        self.remote = path_or_url.startswith(("http://", "https://"))
        self.buf = b""
        self.pos = 0
        self.chunk = chunk
        self.max_bytes = max_bytes
        self._fh = None if self.remote else open(path_or_url, "rb")

    def _fill(self, need: int) -> None:
        while len(self.buf) < need:
            if len(self.buf) >= self.max_bytes:
                raise ValueError("GGUF header larger than limit")
            start = len(self.buf)
            if self.remote:
                req = urllib.request.Request(self.loc, headers={
                    "User-Agent": "tlav", "Range": f"bytes={start}-{start + self.chunk - 1}"})
                tok = os.environ.get("HF_TOKEN")
                if tok:
                    req.add_header("Authorization", f"Bearer {tok}")
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
            else:
                self._fh.seek(start)
                data = self._fh.read(self.chunk)
            if not data:
                raise EOFError("unexpected end of GGUF file")
            self.buf += data

    def read(self, n: int) -> bytes:
        self._fill(self.pos + n)
        out = self.buf[self.pos:self.pos + n]
        self.pos += n
        return out

    def close(self):
        if self._fh:
            self._fh.close()


def _u32(s): return struct.unpack("<I", s.read(4))[0]
def _u64(s): return struct.unpack("<Q", s.read(8))[0]
def _str(s): return s.read(_u64(s)).decode("utf-8", "replace")


def _value(s, vtype: int, keep_arrays: bool) -> Any:
    if vtype in _SCALARS:
        fmt = _SCALARS[vtype]
        return struct.unpack(fmt, s.read(struct.calcsize(fmt)))[0]
    if vtype == _STRING:
        return _str(s)
    if vtype == _ARRAY:
        etype, n = _u32(s), _u64(s)
        if etype in _SCALARS and not keep_arrays and n > 64:
            s.read(struct.calcsize(_SCALARS[etype]) * n)  # skip big numeric arrays (token scores...)
            return None
        items = [_value(s, etype, keep_arrays) for _ in range(n)]
        return items if (keep_arrays or n <= 64) else None
    raise ValueError(f"unknown GGUF value type {vtype}")


@dataclass
class GGUFInfo:
    arch: str
    name: str
    metadata: Dict[str, Any]
    n_params: int
    n_expert_params: int = 0
    tensor_types: Dict[str, int] = field(default_factory=dict)

    def _a(self, key: str, default=None):
        return self.metadata.get(f"{self.arch}.{key}", default)

    @property
    def layers(self) -> int:
        return int(self._a("block_count", 0))

    @property
    def context_length(self) -> int:
        return int(self._a("context_length", 4096))

    @property
    def kv_heads(self) -> int:
        v = self._a("attention.head_count_kv", self._a("attention.head_count", 1))
        if isinstance(v, list):
            v = max(v) if v else 1
        return int(v)

    @property
    def head_dim(self) -> int:
        k = self._a("attention.key_length")
        if k:
            return int(k)
        heads = self._a("attention.head_count", 1)
        if isinstance(heads, list):
            heads = max(heads)
        return int(self._a("embedding_length", 4096)) // max(1, int(heads))

    @property
    def expert_count(self) -> int:
        return int(self._a("expert_count", 0) or 0)

    @property
    def expert_used(self) -> int:
        return int(self._a("expert_used_count", 0) or 0)

    @property
    def active_params(self) -> int:
        if self.expert_count and self.expert_used and self.n_expert_params:
            dense = self.n_params - self.n_expert_params
            return int(dense + self.n_expert_params * self.expert_used / self.expert_count)
        return self.n_params

    def to_model(self, model_id: str, repo: str = ""):
        """Turn header info into a catalog.Model the planner understands."""
        from .catalog import Model, estimate_quality
        pb = self.n_params / 1e9
        ab = self.active_params / 1e9
        q = estimate_quality(ab if self.expert_count else pb, moe_total_b=pb if self.expert_count else None)
        return Model(id=model_id, name=self.name or model_id, repo=repo, params_b=round(pb, 6),
                     layers=self.layers, kv_heads=self.kv_heads, head_dim=self.head_dim,
                     max_ctx=self.context_length, quality={"chat": q, "code": q - 3}, tags=["chat"],
                     active_params_b=round(ab, 6) if self.expert_count else None)


def read(path_or_url: str) -> GGUFInfo:
    s = _Source(path_or_url)
    try:
        if s.read(4) != GGUF_MAGIC:
            raise ValueError("not a GGUF file")
        version = _u32(s)
        if version < 2:
            raise ValueError(f"GGUF v{version} is too old")
        n_tensors, n_kv = _u64(s), _u64(s)
        meta: Dict[str, Any] = {}
        for _ in range(n_kv):
            key = _str(s)
            vtype = _u32(s)
            meta[key] = _value(s, vtype, keep_arrays=False)
        n_params = n_expert = 0
        types: Dict[str, int] = {}
        for _ in range(n_tensors):
            tname = _str(s)
            dims = [_u64(s) for _ in range(_u32(s))]
            ttype = _u32(s)
            _u64(s)  # offset
            count = 1
            for d in dims:
                count *= d
            n_params += count
            if "_exps" in tname:
                n_expert += count
            types[str(ttype)] = types.get(str(ttype), 0) + count
    finally:
        s.close()
    arch = meta.get("general.architecture", "unknown")
    return GGUFInfo(arch=arch, name=meta.get("general.name", ""), metadata=meta, n_params=n_params,
                    n_expert_params=n_expert, tensor_types=types)


def hf_url(repo: str, filename: str) -> str:
    from .hub import HF
    return f"{HF}/{repo}/resolve/main/{filename}"


def read_remote(repo: str, filename: str) -> GGUFInfo:
    return read(hf_url(repo, filename))


def write_minimal(path: str, arch: str, meta: Dict[str, Any], tensors: List[tuple]) -> None:
    """Write a tiny header-only GGUF (used by tests). tensors: [(name, dims, type)]."""
    def s(x: str) -> bytes:
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    out = [GGUF_MAGIC, struct.pack("<I", 3), struct.pack("<Q", len(tensors)),
           struct.pack("<Q", len(meta) + 1), s("general.architecture"), struct.pack("<I", _STRING), s(arch)]
    for k, v in meta.items():
        out.append(s(k))
        if isinstance(v, str):
            out += [struct.pack("<I", _STRING), s(v)]
        elif isinstance(v, list):
            out += [struct.pack("<I", _ARRAY), struct.pack("<I", 4), struct.pack("<Q", len(v))]
            out += [struct.pack("<I", x) for x in v]
        else:
            out += [struct.pack("<I", 4), struct.pack("<I", v)]
    for name, dims, ttype in tensors:
        out += [s(name), struct.pack("<I", len(dims))] + [struct.pack("<Q", d) for d in dims]
        out += [struct.pack("<I", ttype), struct.pack("<Q", 0)]
    with open(path, "wb") as f:
        f.write(b"".join(out))
