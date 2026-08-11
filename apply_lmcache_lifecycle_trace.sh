#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$PWD}"
ROOT="$(cd "$ROOT" && pwd)"

if [[ ! -d "$ROOT/third_party/LMCache/lmcache/v1" ]]; then
  echo "ERROR: $ROOT does not look like the KV-Aware-vLLM repo root" >&2
  exit 2
fi

python3 - "$ROOT" <<'PY'
from __future__ import annotations
from pathlib import Path
import re
import shutil
import sys
import time

ROOT = Path(sys.argv[1]).resolve()
F = {
    "sc": ROOT / "third_party/LMCache/lmcache/v1/sc_config.py",
    "mem": ROOT / "third_party/LMCache/lmcache/v1/memory_management.py",
    "cpu": ROOT / "third_party/LMCache/lmcache/v1/storage_backend/local_cpu_backend.py",
    "disk": ROOT / "third_party/LMCache/lmcache/v1/storage_backend/local_disk_backend.py",
    "storage": ROOT / "third_party/LMCache/lmcache/v1/storage_backend/storage_manager.py",
    "engine": ROOT / "third_party/LMCache/lmcache/v1/cache_engine.py",
    "gpu": ROOT / "third_party/LMCache/lmcache/v1/gpu_connector/gpu_connectors.py",
    "knobs": ROOT / "local_repro/sc_lmcache_knobs.sh",
    "docs": ROOT / "local_repro/SC_LMCACHE_KNOBS.md",
}
for name, path in F.items():
    if not path.exists():
        raise SystemExit(f"ERROR: missing expected {name} file: {path}")

stamp = time.strftime("%Y%m%d_%H%M%S")
backup = ROOT / f".sc_lifecycle_trace_backup_{stamp}"
backed_up: set[Path] = set()
changed: list[Path] = []

def rd(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def wr(p: Path, s: str) -> None:
    if p not in backed_up:
        dst = backup / p.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        backed_up.add(p)
    p.write_text(s, encoding="utf-8")
    if p not in changed:
        changed.append(p)

def repl(p: Path, old: str, new: str, marker: str) -> None:
    s = rd(p)
    if marker in s:
        print(f"SKIP {p.relative_to(ROOT)} :: {marker}")
        return
    n = s.count(old)
    if n != 1:
        raise SystemExit(
            f"ERROR: anchor mismatch in {p.relative_to(ROOT)} for {marker!r}: found {n}"
        )
    wr(p, s.replace(old, new, 1))
    print(f"PATCH {p.relative_to(ROOT)} :: {marker}")

def regex_repl(p: Path, pattern: str, replacement, marker: str) -> None:
    s = rd(p)
    if marker in s:
        print(f"SKIP {p.relative_to(ROOT)} :: {marker}")
        return
    ms = list(re.finditer(pattern, s, re.MULTILINE))
    if len(ms) != 1:
        raise SystemExit(
            f"ERROR: regex anchor mismatch in {p.relative_to(ROOT)} for {marker!r}: "
            f"found {len(ms)}"
        )
    out, n = re.subn(pattern, replacement, s, count=1, flags=re.MULTILINE)
    assert n == 1
    wr(p, out)
    print(f"PATCH {p.relative_to(ROOT)} :: {marker}")

def section_repl(p: Path, start: str, end: str, old: str, new: str, marker: str) -> None:
    s = rd(p)
    if marker in s:
        print(f"SKIP {p.relative_to(ROOT)} :: {marker}")
        return
    a = s.find(start)
    b = s.find(end, a + len(start))
    if a < 0 or b < 0:
        raise SystemExit(f"ERROR: section anchors not found in {p.relative_to(ROOT)}")
    part = s[a:b]
    n = part.count(old)
    if n != 1:
        raise SystemExit(
            f"ERROR: section anchor mismatch in {p.relative_to(ROOT)} for {marker!r}: "
            f"found {n}"
        )
    wr(p, s[:a] + part.replace(old, new, 1) + s[b:])
    print(f"PATCH {p.relative_to(ROOT)} :: {marker}")

# ------------------------------------------------------------------
# sc_config.py
# ------------------------------------------------------------------
regex_repl(
    F["sc"],
    r'(def gpu_assert_snapshot_enabled\(\)\s*->\s*bool:\n\s+return env_flag\("SC_LMCACHE_GPU_ASSERT_SNAPSHOT_ENABLE"\)\n)',
    lambda m: m.group(1)
    + '\n\ndef lifecycle_trace_enabled() -> bool:\n'
      '    return env_flag("SC_LMCACHE_LIFECYCLE_TRACE_ENABLE")\n',
    "def lifecycle_trace_enabled",
)

# ------------------------------------------------------------------
# memory_management.py: common logger + fundamental transitions
# ------------------------------------------------------------------
repl(
    F["mem"],
    """from lmcache.v1.sc_config import (\n    env_float,\n    memory_snapshot_enabled,\n    memory_trace_enabled,\n)\n""",
    """from lmcache.v1.sc_config import (\n    env_float,\n    lifecycle_trace_enabled,\n    memory_snapshot_enabled,\n    memory_trace_enabled,\n)\n""",
    "lifecycle_trace_enabled,",
)
repl(
    F["mem"],
    "import os as _sc_os\nimport time as _sc_time\n",
    "import os as _sc_os\nimport sys as _sc_sys\nimport time as _sc_time\n",
    "import sys as _sc_sys",
)
helper = '''# ===== SC MEMORY LIFECYCLE TRACE START =====
def _sc_lifecycle_callsite():
    try:
        # actual caller -> MemoryObj method -> trace helper -> here
        frame = _sc_sys._getframe(3)
        return (
            f"{_sc_os.path.basename(frame.f_code.co_filename)}:"
            f"{frame.f_lineno}:{frame.f_code.co_name}"
        )
    except Exception:
        return "unknown"


def _sc_memory_lifecycle_trace(
    event,
    obj,
    *,
    lookup_id="",
    key=None,
    backend="",
    site="",
    before_ref=None,
    before_pin=None,
    extra="",
):
    if not lifecycle_trace_enabled():
        return
    try:
        meta = getattr(obj, "meta", None)
        key_hash = getattr(key, "chunk_hash", None) if key is not None else None
        if not site:
            site = _sc_lifecycle_callsite()
        logger.warning(
            "[SC_LIFECYCLE] "
            "event=%s mono_ns=%d pid=%d thread=%s "
            "obj_id=%d address=%s valid=%s "
            "ref=%s pin=%s before_ref=%s before_pin=%s "
            "lookup_id=%s key_hash=%s backend=%s site=%s extra=%s",
            event,
            _sc_time.monotonic_ns(),
            _sc_os.getpid(),
            threading.current_thread().name,
            id(obj),
            getattr(meta, "address", None),
            getattr(obj, "valid", None),
            getattr(meta, "ref_count", None),
            getattr(meta, "pin_count", None),
            before_ref,
            before_pin,
            lookup_id,
            key_hash,
            backend,
            site,
            extra,
        )
    except Exception:
        logger.exception("[SC_LIFECYCLE_ERROR] event=%s", event)
# ===== SC MEMORY LIFECYCLE TRACE END =====
'''
repl(
    F["mem"],
    "# Helper functions for thread safety\n",
    helper + "\n\n# Helper functions for thread safety\n",
    "SC MEMORY LIFECYCLE TRACE START",
)
section_repl(
    F["mem"], "class TensorMemoryObj(MemoryObj):", "class BytesBufferMemoryObj(MemoryObj):",
    """    def invalidate(self):\n        self.valid = False\n""",
    """    def invalidate(self):\n        self.valid = False\n        _sc_memory_lifecycle_trace("INVALIDATE", self)\n""",
    '_sc_memory_lifecycle_trace("INVALIDATE", self)',
)
section_repl(
    F["mem"], "class TensorMemoryObj(MemoryObj):", "class BytesBufferMemoryObj(MemoryObj):",
    """    def ref_count_up(self):\n        with self.lock:\n            if (\n                self.meta.ref_count == 1\n                and self._sc_ref_gt1_started is None\n            ):\n                self._sc_ref_gt1_started = _sc_time.monotonic()\n            self.meta.ref_count += 1\n""",
    """    def ref_count_up(self):\n        with self.lock:\n            before_ref = self.meta.ref_count\n            if (\n                self.meta.ref_count == 1\n                and self._sc_ref_gt1_started is None\n            ):\n                self._sc_ref_gt1_started = _sc_time.monotonic()\n            self.meta.ref_count += 1\n            _sc_memory_lifecycle_trace(\n                "REF_UP", self, before_ref=before_ref\n            )\n""",
    '"REF_UP", self',
)
section_repl(
    F["mem"], "class TensorMemoryObj(MemoryObj):", "class BytesBufferMemoryObj(MemoryObj):",
    """    def ref_count_down(self):\n        with self.lock:\n            self.meta.ref_count -= 1\n\n""",
    """    def ref_count_down(self):\n        with self.lock:\n            before_ref = self.meta.ref_count\n            self.meta.ref_count -= 1\n            _sc_memory_lifecycle_trace(\n                "REF_DOWN", self, before_ref=before_ref\n            )\n\n""",
    '"REF_DOWN", self',
)
section_repl(
    F["mem"], "class TensorMemoryObj(MemoryObj):", "class BytesBufferMemoryObj(MemoryObj):",
    """    def pin(self) -> bool:\n        with self.lock:\n            # if pin_count is 0, indicates that the object is pinned for the first time\n            if self.meta.pin_count == 0:\n                TensorMemoryObj.monitor.update_pinned_memory_objs_count(1)\n                self._sc_pin_started = _sc_time.monotonic()\n\n            self.meta.pin_count += 1\n\n""",
    """    def pin(self) -> bool:\n        with self.lock:\n            before_pin = self.meta.pin_count\n            # if pin_count is 0, indicates that the object is pinned for the first time\n            if self.meta.pin_count == 0:\n                TensorMemoryObj.monitor.update_pinned_memory_objs_count(1)\n                self._sc_pin_started = _sc_time.monotonic()\n\n            self.meta.pin_count += 1\n            _sc_memory_lifecycle_trace(\n                "PIN", self, before_pin=before_pin\n            )\n\n""",
    '"PIN", self',
)
section_repl(
    F["mem"], "class TensorMemoryObj(MemoryObj):", "class BytesBufferMemoryObj(MemoryObj):",
    """    def unpin(self) -> bool:\n        with self.lock:\n            self.meta.pin_count -= 1\n\n""",
    """    def unpin(self) -> bool:\n        with self.lock:\n            before_pin = self.meta.pin_count\n            self.meta.pin_count -= 1\n            _sc_memory_lifecycle_trace(\n                "UNPIN", self, before_pin=before_pin\n            )\n\n""",
    '"UNPIN", self',
)
section_repl(
    F["mem"],
    "class TensorMemoryAllocator(MemoryAllocatorInterface):",
    "class PagedTensorMemoryAllocator(MemoryAllocatorInterface):",
    """    def free(self, memory_obj: MemoryObj, allocator_type: Optional[str] = None):\n        if not memory_obj.is_valid():\n            return\n\n        self.address_manager.free(memory_obj.meta.address, memory_obj.meta.phy_size)\n        memory_obj.invalidate()\n""",
    """    def free(self, memory_obj: MemoryObj, allocator_type: Optional[str] = None):\n        if not memory_obj.is_valid():\n            return\n\n        _sc_memory_lifecycle_trace(\n            "ALLOCATOR_FREE_ENTER", memory_obj, site="TensorMemoryAllocator.free"\n        )\n        self.address_manager.free(memory_obj.meta.address, memory_obj.meta.phy_size)\n        memory_obj.invalidate()\n        _sc_memory_lifecycle_trace(\n            "ALLOCATOR_FREE_DONE", memory_obj, site="TensorMemoryAllocator.free"\n        )\n""",
    '"ALLOCATOR_FREE_ENTER", memory_obj',
)
section_repl(
    F["mem"],
    "class TensorMemoryAllocator(MemoryAllocatorInterface):",
    "class PagedTensorMemoryAllocator(MemoryAllocatorInterface):",
    """        for memory_obj in memory_objs:\n            if not memory_obj.is_valid():\n                logger.warning("Trying to free an invalidated MemoryObj")\n                continue\n            memory_obj.invalidate()\n""",
    """        for memory_obj in memory_objs:\n            if not memory_obj.is_valid():\n                logger.warning("Trying to free an invalidated MemoryObj")\n                continue\n            _sc_memory_lifecycle_trace(\n                "ALLOCATOR_BATCH_FREE_ENTER",\n                memory_obj,\n                site="TensorMemoryAllocator.batched_free",\n            )\n            memory_obj.invalidate()\n""",
    '"ALLOCATOR_BATCH_FREE_ENTER",',
)

# ------------------------------------------------------------------
# LocalCPUBackend: bind lookup_id/key to object identity
# ------------------------------------------------------------------
repl(
    F["cpu"],
    """    MemoryObj,\n    MixedMemoryAllocator,\n""",
    """    MemoryObj,\n    MixedMemoryAllocator,\n    _sc_memory_lifecycle_trace,\n""",
    "_sc_memory_lifecycle_trace,",
)
repl(
    F["cpu"],
    """                if pin:\n                    self.hot_cache[key].pin()\n                    # vllm lookup sets pin to True\n                    self.keys_in_request.append(key)\n""",
    """                if pin:\n                    self.hot_cache[key].pin()\n                    _sc_memory_lifecycle_trace(\n                        "CPU_LOOKUP_PIN",\n                        self.hot_cache[key],\n                        lookup_id=lookup_id,\n                        key=key,\n                        backend="LocalCPUBackend",\n                        site="LocalCPUBackend.batched_async_contains",\n                    )\n                    # vllm lookup sets pin to True\n                    self.keys_in_request.append(key)\n""",
    '"CPU_LOOKUP_PIN",',
)
repl(
    F["cpu"],
    """            for key in keys:\n                mem_obj = self.hot_cache[key]\n                mem_obj.ref_count_up()\n                mem_objs.append(mem_obj)\n""",
    """            for key in keys:\n                mem_obj = self.hot_cache[key]\n                mem_obj.ref_count_up()\n                _sc_memory_lifecycle_trace(\n                    "CPU_LOOKUP_GET_REF",\n                    mem_obj,\n                    lookup_id=lookup_id,\n                    key=key,\n                    backend="LocalCPUBackend",\n                    site="LocalCPUBackend.batched_get_non_blocking",\n                )\n                mem_objs.append(mem_obj)\n""",
    '"CPU_LOOKUP_GET_REF",',
)
repl(
    F["cpu"],
    """            memory_obj = self.hot_cache[key]\n            memory_obj.unpin()\n            return True\n""",
    """            memory_obj = self.hot_cache[key]\n            _sc_memory_lifecycle_trace(\n                "CPU_KEY_UNPIN",\n                memory_obj,\n                key=key,\n                backend="LocalCPUBackend",\n                site="LocalCPUBackend.unpin",\n            )\n            memory_obj.unpin()\n            return True\n""",
    '"CPU_KEY_UNPIN",',
)
repl(
    F["cpu"],
    """        memory_obj = self.hot_cache.pop(key)\n        memory_obj.ref_count_down()\n""",
    """        memory_obj = self.hot_cache.pop(key)\n        _sc_memory_lifecycle_trace(\n            "CPU_CACHE_REMOVE",\n            memory_obj,\n            key=key,\n            backend="LocalCPUBackend",\n            site="LocalCPUBackend.remove",\n        )\n        memory_obj.ref_count_down()\n""",
    '"CPU_CACHE_REMOVE",',
)

# ------------------------------------------------------------------
# LocalDiskBackend: disk-put ownership
# ------------------------------------------------------------------
repl(
    F["disk"],
    "from lmcache.v1.memory_management import MemoryFormat, MemoryObj\n",
    """from lmcache.v1.memory_management import (\n    MemoryFormat,\n    MemoryObj,\n    _sc_memory_lifecycle_trace,\n)\n""",
    "_sc_memory_lifecycle_trace,",
)
repl(
    F["disk"],
    """        # This extra ref is now taken only after bounded admission succeeds.\n        memory_obj.ref_count_up()\n\n        try:\n""",
    """        # This extra ref is now taken only after bounded admission succeeds.\n        memory_obj.ref_count_up()\n        _sc_memory_lifecycle_trace(\n            "DISK_PUT_REF_UP",\n            memory_obj,\n            key=key,\n            backend="LocalDiskBackend",\n            site="LocalDiskBackend.submit_put_task",\n        )\n\n        try:\n""",
    '"DISK_PUT_REF_UP",',
)
repl(
    F["disk"],
    """        cached_positions = memory_obj.metadata.cached_positions\n        memory_obj.ref_count_down()\n\n        self.insert_key(key, size, shape, dtype, fmt, cached_positions=cached_positions)\n""",
    """        cached_positions = memory_obj.metadata.cached_positions\n        _sc_memory_lifecycle_trace(\n            "DISK_PUT_REF_DOWN",\n            memory_obj,\n            key=key,\n            backend="LocalDiskBackend",\n            site="LocalDiskBackend.async_save_bytes_to_disk",\n        )\n        memory_obj.ref_count_down()\n\n        self.insert_key(key, size, shape, dtype, fmt, cached_positions=cached_positions)\n""",
    '"DISK_PUT_REF_DOWN",',
)

# ------------------------------------------------------------------
# StorageManager: successful lookup result becomes READY
# ------------------------------------------------------------------
repl(
    F["storage"],
    """    MemoryFormat,\n    MemoryObj,\n)\n""",
    """    MemoryFormat,\n    MemoryObj,\n    _sc_memory_lifecycle_trace,\n)\n""",
    "_sc_memory_lifecycle_trace,",
)
repl(
    F["storage"],
    """        retrieved_length = cum_chunk_lengths_total[total_retrieved_chunks]\n        logger.info(\n""",
    """        retrieved_length = cum_chunk_lengths_total[total_retrieved_chunks]\n\n        remaining_ready_keys = total_retrieved_chunks * keys_per_chunk\n        for tier_idx, tier_result in enumerate(res):\n            if remaining_ready_keys <= 0:\n                break\n            ready_slice = tier_result[:remaining_ready_keys]\n            for key, memory_obj in ready_slice:\n                _sc_memory_lifecycle_trace(\n                    "LOOKUP_READY",\n                    memory_obj,\n                    lookup_id=lookup_id,\n                    key=key,\n                    backend="StorageManager",\n                    site="StorageManager.prefetch_all_done_callback",\n                    extra=f"tier={tier_idx} retrieved_length={retrieved_length}",\n                )\n            remaining_ready_keys -= len(ready_slice)\n\n        logger.info(\n""",
    '"LOOKUP_READY",',
)

# ------------------------------------------------------------------
# LMCacheEngine: READY event -> retrieve claim; abort cleanup
# ------------------------------------------------------------------
repl(
    F["engine"],
    """    TensorMemoryObj,\n)\n""",
    """    TensorMemoryObj,\n    _sc_memory_lifecycle_trace,\n)\n""",
    "_sc_memory_lifecycle_trace,",
)
repl(
    F["engine"],
    """        for backend_results in keyed_memory_objs:\n            for key, memory_obj in backend_results:\n                memory_obj_map[key] = memory_obj\n""",
    """        for backend_results in keyed_memory_objs:\n            for key, memory_obj in backend_results:\n                _sc_memory_lifecycle_trace(\n                    "RETRIEVE_EVENT_READ",\n                    memory_obj,\n                    lookup_id=kwargs["req_id"],\n                    key=key,\n                    backend="LMCacheEngine",\n                    site="LMCacheEngine._async_process_tokens_internal",\n                )\n                memory_obj_map[key] = memory_obj\n""",
    '"RETRIEVE_EVENT_READ",',
)
repl(
    F["engine"],
    """            chunks.append((key, memory_obj, start, end))\n            tot_kv_size += memory_obj.get_size()\n""",
    """            _sc_memory_lifecycle_trace(\n                "RETRIEVE_CLAIM",\n                memory_obj,\n                lookup_id=kwargs["req_id"],\n                key=key,\n                backend="LMCacheEngine",\n                site="LMCacheEngine._async_process_tokens_internal",\n                extra=f"start={start} end={end}",\n            )\n            chunks.append((key, memory_obj, start, end))\n            tot_kv_size += memory_obj.get_size()\n""",
    '"RETRIEVE_CLAIM",',
)
repl(
    F["engine"],
    """        for key, mem_obj in memory_obj_map.items():\n            if key not in used_keys:\n                mem_obj.ref_count_down()\n""",
    """        for key, mem_obj in memory_obj_map.items():\n            if key not in used_keys:\n                _sc_memory_lifecycle_trace(\n                    "RETRIEVE_UNUSED_REF_DOWN",\n                    mem_obj,\n                    lookup_id=kwargs["req_id"],\n                    key=key,\n                    backend="LMCacheEngine",\n                    site="LMCacheEngine._async_process_tokens_internal",\n                )\n                mem_obj.ref_count_down()\n""",
    '"RETRIEVE_UNUSED_REF_DOWN",',
)
repl(
    F["engine"],
    """            for key, memory_obj in memory_objs_flat:\n                try:\n                    logger.debug("Releasing memory object for lookup_id=%s", lookup_id)\n                    if memory_obj.is_pinned:\n""",
    """            for key, memory_obj in memory_objs_flat:\n                try:\n                    logger.debug("Releasing memory object for lookup_id=%s", lookup_id)\n                    _sc_memory_lifecycle_trace(\n                        "ABORT_CLEANUP_RELEASE",\n                        memory_obj,\n                        lookup_id=lookup_id,\n                        key=key,\n                        backend="LMCacheEngine",\n                        site="LMCacheEngine.cleanup_memory_objs",\n                    )\n                    if memory_obj.is_pinned:\n""",
    '"ABORT_CLEANUP_RELEASE",',
)

# ------------------------------------------------------------------
# GPU connector: pre-assert state and H2D completion
# ------------------------------------------------------------------
repl(
    F["gpu"],
    "from lmcache.v1.memory_management import MemoryFormat, MemoryObj\n",
    """from lmcache.v1.memory_management import (\n    MemoryFormat,\n    MemoryObj,\n    _sc_memory_lifecycle_trace,\n)\n""",
    "_sc_memory_lifecycle_trace,",
)
# Scope this insertion to VLLMPagedMemGPUConnectorV2.to_gpu to avoid matching
# other connector docstrings.
s = rd(F["gpu"])
if '"GPU_TO_GPU_ENTER",' not in s:
    class_pos = s.find("class VLLMPagedMemGPUConnectorV2")
    method_pos = s.find("    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):", class_pos)
    tensor_pos = s.find("        if memory_obj.tensor is None:\n", method_pos)
    if class_pos < 0 or method_pos < 0 or tensor_pos < 0:
        raise SystemExit("ERROR: could not locate VLLMPagedMemGPUConnectorV2.to_gpu tensor check")
    insert = '''        _sc_memory_lifecycle_trace(\n            "GPU_TO_GPU_ENTER",\n            memory_obj,\n            lookup_id=str(kwargs.get("req_id", "")),\n            backend="VLLMPagedMemGPUConnectorV2",\n            site="VLLMPagedMemGPUConnectorV2.to_gpu",\n            extra=f"start={start} end={end}",\n        )\n'''
    wr(F["gpu"], s[:tensor_pos] + insert + s[tensor_pos:])
    print(f"PATCH {F['gpu'].relative_to(ROOT)} :: GPU_TO_GPU_ENTER")
else:
    print(f"SKIP {F['gpu'].relative_to(ROOT)} :: GPU_TO_GPU_ENTER")
section_repl(
    F["gpu"],
    "class VLLMPagedMemGPUConnectorV2",
    "class VLLMPagedMemGPUConnectorV3",
    """        with torch.cuda.stream(self.load_stream):\n            for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):\n                self.to_gpu(memory_obj, start, end, **kwargs)\n        self.load_stream.synchronize()\n""",
    """        with torch.cuda.stream(self.load_stream):\n            for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):\n                self.to_gpu(memory_obj, start, end, **kwargs)\n        self.load_stream.synchronize()\n        for memory_obj in memory_objs:\n            _sc_memory_lifecycle_trace(\n                "GPU_COPY_DONE",\n                memory_obj,\n                lookup_id=str(kwargs.get("req_id", "")),\n                backend="VLLMPagedMemGPUConnectorV2",\n                site="VLLMPagedMemGPUConnectorV2.batched_to_gpu",\n            )\n""",
    '"GPU_COPY_DONE",',
)

# ------------------------------------------------------------------
# knob plumbing + docs
# ------------------------------------------------------------------
repl(
    F["knobs"],
    """_sc_default SC_LMCACHE_GPU_ASSERT_SNAPSHOT_ENABLE 0\n\n_sc_default SC_LMCACHE_MEMORY_TRACE_INTERVAL_S 5\n""",
    """_sc_default SC_LMCACHE_GPU_ASSERT_SNAPSHOT_ENABLE 0\n# Extremely verbose per-MemoryObj ownership trace; always opt-in.\n_sc_default SC_LMCACHE_LIFECYCLE_TRACE_ENABLE 0\n\n_sc_default SC_LMCACHE_MEMORY_TRACE_INTERVAL_S 5\n""",
    "_sc_default SC_LMCACHE_LIFECYCLE_TRACE_ENABLE",
)
repl(
    F["knobs"],
    """  SC_LMCACHE_TIER_TRACE_ENABLE \\\n  SC_LMCACHE_GPU_ASSERT_SNAPSHOT_ENABLE \\\n  SC_DRIVER_RESOURCE_MONITOR_ENABLE; do\n""",
    """  SC_LMCACHE_TIER_TRACE_ENABLE \\\n  SC_LMCACHE_GPU_ASSERT_SNAPSHOT_ENABLE \\\n  SC_LMCACHE_LIFECYCLE_TRACE_ENABLE \\\n  SC_DRIVER_RESOURCE_MONITOR_ENABLE; do\n""",
    "  SC_LMCACHE_LIFECYCLE_TRACE_ENABLE \\\n  SC_DRIVER_RESOURCE_MONITOR_ENABLE; do",
)
repl(
    F["knobs"],
    """    SC_LMCACHE_TIER_TRACE_ENABLE \\\n    SC_LMCACHE_GPU_ASSERT_SNAPSHOT_ENABLE \\\n    SC_LMCACHE_MEMORY_TRACE_INTERVAL_S \\\n""",
    """    SC_LMCACHE_TIER_TRACE_ENABLE \\\n    SC_LMCACHE_GPU_ASSERT_SNAPSHOT_ENABLE \\\n    SC_LMCACHE_LIFECYCLE_TRACE_ENABLE \\\n    SC_LMCACHE_MEMORY_TRACE_INTERVAL_S \\\n""",
    "    SC_LMCACHE_LIFECYCLE_TRACE_ENABLE \\\n    SC_LMCACHE_MEMORY_TRACE_INTERVAL_S",
)
repl(
    F["docs"],
    """| `SC_LMCACHE_GPU_ASSERT_SNAPSHOT_ENABLE` | `0` | Heavy snapshot immediately before the GPU connector tensor assertion; independent of the general memory-snapshot flag. |\n| `SC_DRIVER_RESOURCE_MONITOR_ENABLE` | `0` | Periodic driver-level process/GPU/disk monitor. |\n""",
    """| `SC_LMCACHE_GPU_ASSERT_SNAPSHOT_ENABLE` | `0` | Heavy snapshot immediately before the GPU connector tensor assertion; independent of the general memory-snapshot flag. |\n| `SC_LMCACHE_LIFECYCLE_TRACE_ENABLE` | `0` | Very verbose per-`MemoryObj` lookup/ref/pin/free/invalidation/GPU-consumption lifecycle trace for targeted debugging. |\n| `SC_DRIVER_RESOURCE_MONITOR_ENABLE` | `0` | Periodic driver-level process/GPU/disk monitor. |\n""",
    "`SC_LMCACHE_LIFECYCLE_TRACE_ENABLE`",
)

# ------------------------------------------------------------------
# Convenience environment file. Not sourced automatically.
# ------------------------------------------------------------------
env_file = ROOT / "local_repro/lifecycle_debug_run00_env.sh"
if not env_file.exists():
    env_file.write_text('''#!/usr/bin/env bash
# Source this before submitting/running the targeted lifecycle reproduction.
# Reproduces round-2 run 00; intentionally leaves SC_LMCACHE_DATA_DIR unset.

export MAX_QUESTIONS=250
export DATASET_NAME=medical
export SUBMISSION_BATCH_SIZE=50
export VLLM_MAX_NUM_SEQS=12
export VLLM_KV_IMPORTANCE_ENABLE=0

export SC_LMCACHE_SOURCE_DEFAULT_KNOBS=1
export SC_LMCACHE_PROFILE=custom
export SC_LMCACHE_SCHEDULER_LOOKUP_ADMISSION_ENABLE=1
export SC_LMCACHE_SCHEDULER_LOOKUP_MAX_INFLIGHT=4
export SC_LMCACHE_SCHEDULER_LOOKUP_ADMISSION_WAIT_MS=0
export SC_LMCACHE_WORKER_LOOKUP_ADMISSION_ENABLE=0
export SC_LMCACHE_WORKER_LOOKUP_MAX_INFLIGHT=1
export SC_LMCACHE_DISK_PUT_ADMISSION_ENABLE=0
export SC_LMCACHE_DISK_PUT_MAX_PENDING=8

export SC_LMCACHE_IO_TRACE_ENABLE=1
export SC_LMCACHE_LOAD_TRACE_ENABLE=1
export SC_LMCACHE_MEMORY_TRACE_ENABLE=1
export SC_LMCACHE_MEMORY_SNAPSHOT_ENABLE=0
export SC_LMCACHE_LOOKUP_TRACE_ENABLE=0
export SC_LMCACHE_REQUEST_TRACE_ENABLE=0
export SC_LMCACHE_TIER_TRACE_ENABLE=0
export SC_LMCACHE_LIFECYCLE_TRACE_ENABLE=1
export SC_LMCACHE_GPU_ASSERT_SNAPSHOT_ENABLE=1

export SC_LMCACHE_MEMORY_TRACE_INTERVAL_S=5
export SC_LMCACHE_LONG_PIN_THRESHOLD_S=10
export SC_LMCACHE_LONG_REF_THRESHOLD_S=10
export SC_DRIVER_RESOURCE_MONITOR_ENABLE=1
export SC_DRIVER_RESOURCE_MONITOR_INTERVAL_S=30
export CLEAN_OLD_LMCACHE=1
''', encoding="utf-8")
    env_file.chmod(0o755)
    print(f"CREATE {env_file.relative_to(ROOT)}")
else:
    print(f"SKIP existing {env_file.relative_to(ROOT)}")

print("\nPatch complete.")
print(f"Backups: {backup}")
for p in changed:
    print(f"  changed: {p.relative_to(ROOT)}")
PY

# Validate syntax. These checks do not import torch/vLLM/LMCache.
PYFILES=(
  "$ROOT/third_party/LMCache/lmcache/v1/sc_config.py"
  "$ROOT/third_party/LMCache/lmcache/v1/memory_management.py"
  "$ROOT/third_party/LMCache/lmcache/v1/storage_backend/local_cpu_backend.py"
  "$ROOT/third_party/LMCache/lmcache/v1/storage_backend/local_disk_backend.py"
  "$ROOT/third_party/LMCache/lmcache/v1/storage_backend/storage_manager.py"
  "$ROOT/third_party/LMCache/lmcache/v1/cache_engine.py"
  "$ROOT/third_party/LMCache/lmcache/v1/gpu_connector/gpu_connectors.py"
)
python3 -m py_compile "${PYFILES[@]}"
bash -n "$ROOT/local_repro/sc_lmcache_knobs.sh"
bash -n "$ROOT/local_repro/lifecycle_debug_run00_env.sh"

if git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git -C "$ROOT" diff --check
fi

echo
echo "Validation passed."
echo "Lifecycle tracing remains OFF by default."
echo
echo "For the targeted run-00 reproduction:"
echo "  cd '$ROOT'"
echo "  source local_repro/lifecycle_debug_run00_env.sh"
echo "  # optionally export SC_LMCACHE_DATA_DIR=..."
echo "  # then submit/run through your normal run_driver/sbatch path"
echo
echo "Useful grep after the run:"
echo "  grep '\[SC_LIFECYCLE\]' <slurm-log> > lifecycle.log"
echo "  grep 'GPU_TO_GPU_ENTER.*valid=False\|INVALIDATE\|ALLOCATOR_FREE' lifecycle.log"
