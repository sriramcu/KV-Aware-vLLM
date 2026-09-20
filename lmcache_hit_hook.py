import atexit
import functools
import glob
import json
import os
import threading
import time
from typing import Any

_LOG_DIR = os.environ.get("LMCACHE_HOOK_LOG_DIR", "/tmp/lmcache_hit_hook")
_LOCK = threading.Lock()


def _ensure_dir() -> None:
    os.makedirs(_LOG_DIR, exist_ok=True)


def _log_path(prefix: str) -> str:
    _ensure_dir()
    return os.path.join(_LOG_DIR, f"{prefix}_{os.getpid()}.jsonl")


def _append_jsonl(prefix: str, record: dict[str, Any]) -> None:
    path = _log_path(prefix)
    with _LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def reset_logs() -> None:
    _ensure_dir()
    for p in glob.glob(os.path.join(_LOG_DIR, "*.jsonl")):
        try:
            os.remove(p)
        except FileNotFoundError:
            pass


def read_logs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in glob.glob(os.path.join(_LOG_DIR, "*.jsonl")):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def summarize() -> dict[str, Any]:
    rows = read_logs()
    req = {}
    total_new_matched = 0
    total_calls = 0

    for r in rows:
        if r.get("event") not in ("vllm_connector_hit", "lmcache_adapter_hit"):
            continue
        total_calls += 1
        n = r.get("new_matched_tokens")
        if isinstance(n, int) and n > 0:
            total_new_matched += n
        rid = r.get("request_id", "unknown")
        req.setdefault(rid, {"max_new_matched_tokens": 0, "calls": 0, "sites": set()})
        req[rid]["calls"] += 1
        req[rid]["sites"].add(r.get("site"))
        if isinstance(n, int) and n > req[rid]["max_new_matched_tokens"]:
            req[rid]["max_new_matched_tokens"] = n

    # make JSON-serializable
    per_request = {
        rid: {
            "max_new_matched_tokens": info["max_new_matched_tokens"],
            "calls": info["calls"],
            "sites": sorted(info["sites"]),
        }
        for rid, info in req.items()
    }

    return {
        "log_dir": _LOG_DIR,
        "total_rows": len(rows),
        "total_calls": total_calls,
        "sum_new_matched_tokens_over_calls": total_new_matched,
        "per_request": per_request,
    }


def _safe_request_id(request: Any) -> str:
    for attr in ("request_id", "req_id", "id"):
        if hasattr(request, attr):
            try:
                v = getattr(request, attr)
                if v is not None:
                    return str(v)
            except Exception:
                pass
    return "unknown"


def _safe_prompt_len(request: Any) -> int | None:
    for attr in ("prompt_token_ids", "all_token_ids"):
        if hasattr(request, attr):
            try:
                v = getattr(request, attr)
                if v is not None:
                    return len(v)
            except Exception:
                pass
    return None


_PATCHED = False


def install() -> None:
    global _PATCHED
    if _PATCHED:
        return

    # Patch vLLM connector
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
            LMCacheConnectorV1,
        )

        orig_vllm = LMCacheConnectorV1.get_num_new_matched_tokens

        @functools.wraps(orig_vllm)
        def wrapped_vllm(self, request, num_computed_tokens):
            t0 = time.time()
            exc = None
            result = None
            try:
                result = orig_vllm(self, request, num_computed_tokens)
                return result
            except Exception as e:
                exc = repr(e)
                raise
            finally:
                # vLLM method returns tuple[int | None, bool] per API docs
                new_matched_tokens = None
                async_flag = None
                if isinstance(result, tuple) and len(result) >= 2:
                    new_matched_tokens = result[0]
                    async_flag = result[1]
                elif isinstance(result, int):
                    new_matched_tokens = result

                _append_jsonl(
                    "hits",
                    {
                        "ts": time.time(),
                        "site": "vllm_connector",
                        "event": "vllm_connector_hit",
                        "pid": os.getpid(),
                        "request_id": _safe_request_id(request),
                        "prompt_len": _safe_prompt_len(request),
                        "num_computed_tokens": num_computed_tokens,
                        "new_matched_tokens": new_matched_tokens,
                        "async_flag": async_flag,
                        "latency_ms": (time.time() - t0) * 1000.0,
                        "exception": exc,
                    },
                )

        LMCacheConnectorV1.get_num_new_matched_tokens = wrapped_vllm
        _append_jsonl(
            "meta",
            {
                "event": "patch_installed",
                "site": "vllm_connector",
                "pid": os.getpid(),
                "ts": time.time(),
            },
        )
    except Exception as e:
        _append_jsonl(
            "meta",
            {
                "event": "patch_failed",
                "site": "vllm_connector",
                "pid": os.getpid(),
                "ts": time.time(),
                "exception": repr(e),
            },
        )

    # Patch LMCache adapter too, when available
    try:
        from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl

        orig_adapter = LMCacheConnectorV1Impl.get_num_new_matched_tokens

        @functools.wraps(orig_adapter)
        def wrapped_adapter(self, request, num_computed_tokens):
            t0 = time.time()
            exc = None
            result = None
            try:
                result = orig_adapter(self, request, num_computed_tokens)
                return result
            except Exception as e:
                exc = repr(e)
                raise
            finally:
                _append_jsonl(
                    "hits",
                    {
                        "ts": time.time(),
                        "site": "lmcache_adapter",
                        "event": "lmcache_adapter_hit",
                        "pid": os.getpid(),
                        "request_id": _safe_request_id(request),
                        "prompt_len": _safe_prompt_len(request),
                        "num_computed_tokens": num_computed_tokens,
                        "new_matched_tokens": result if isinstance(result, int) else None,
                        "latency_ms": (time.time() - t0) * 1000.0,
                        "exception": exc,
                    },
                )

        LMCacheConnectorV1Impl.get_num_new_matched_tokens = wrapped_adapter
        _append_jsonl(
            "meta",
            {
                "event": "patch_installed",
                "site": "lmcache_adapter",
                "pid": os.getpid(),
                "ts": time.time(),
            },
        )
    except Exception as e:
        _append_jsonl(
            "meta",
            {
                "event": "patch_failed",
                "site": "lmcache_adapter",
                "pid": os.getpid(),
                "ts": time.time(),
                "exception": repr(e),
            },
        )

    _PATCHED = True


def dump_summary(path: str | None = None) -> str:
    summary = summarize()
    if path is None:
        path = os.path.join(_LOG_DIR, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return path


@atexit.register
def _atexit_summary():
    try:
        dump_summary()
    except Exception:
        pass