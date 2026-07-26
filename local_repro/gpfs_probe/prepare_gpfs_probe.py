#!/usr/bin/env python3
"""Create immutable 80 MiB files on GPFS for gpfs_mixed_io_probe.py."""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
MIB = 1024 * 1024

def log(event: str, **fields: object) -> None:
    print(json.dumps({"event": event, "wall": time.time(), "mono": time.monotonic(), **fields}, sort_keys=True), flush=True)

def write_all(fd: int, payload: bytes) -> int:
    view = memoryview(payload); total = 0
    while total < len(view):
        n = os.write(fd, view[total:])
        if n <= 0: raise OSError(f"os.write returned {n}")
        total += n
    return total

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/mnt/shared/gpfs/home/sriramc2/runs/gpfs_io_probe")
    p.add_argument("--num-files", type=int, default=128)
    p.add_argument("--file-mib", type=int, default=80)
    p.add_argument("--chunk-mib", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    if a.file_mib <= 0 or a.chunk_mib <= 0 or a.file_mib % a.chunk_mib:
        raise ValueError("positive sizes required and --file-mib must be divisible by --chunk-mib")
    root = Path(a.root); reader_dir = root / "reader_files"; reader_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    file_size = a.file_mib * MIB; chunk_size = a.chunk_mib * MIB
    payload = os.urandom(chunk_size)
    log("prep_start", root=str(root), num_files=a.num_files, file_size=file_size, total_bytes=a.num_files*file_size)
    files: list[str] = []; total_start = time.monotonic()
    for i in range(a.num_files):
        path = reader_dir / f"reader_{i:05d}.bin"; files.append(str(path))
        if path.exists() and not a.overwrite:
            size = path.stat().st_size
            if size != file_size: raise RuntimeError(f"{path} has {size}, expected {file_size}; use --overwrite")
            log("prep_reuse", index=i, path=str(path), size=size); continue
        tmp = path.with_suffix(".tmp"); tmp.unlink(missing_ok=True); fd = -1
        try:
            t = time.monotonic(); fd = os.open(tmp, os.O_CREAT|os.O_TRUNC|os.O_WRONLY, 0o644); open_s = time.monotonic()-t
            t = time.monotonic(); written = 0
            for _ in range(file_size//chunk_size): written += write_all(fd, payload)
            write_s = time.monotonic()-t
            t = time.monotonic(); os.fdatasync(fd); sync_s = time.monotonic()-t
            t = time.monotonic(); os.close(fd); fd = -1; close_s = time.monotonic()-t
            os.replace(tmp, path); elapsed = open_s+write_s+sync_s+close_s
            log("prep_file_done", index=i, path=str(path), bytes=written, open_s=open_s, write_s=write_s, fdatasync_s=sync_s, close_s=close_s, elapsed_s=elapsed, mib_per_s=written/MIB/elapsed if elapsed else None)
        finally:
            if fd >= 0: os.close(fd)
            tmp.unlink(missing_ok=True)
    manifest = {"version":1,"created_wall":time.time(),"root":str(root),"file_size":file_size,"num_files":len(files),"files":files}
    tmpm = manifest_path.with_suffix(".tmp"); tmpm.write_text(json.dumps(manifest, indent=2)+"\n"); os.replace(tmpm, manifest_path)
    elapsed = time.monotonic()-total_start
    log("prep_done", manifest=str(manifest_path), files=len(files), total_bytes=len(files)*file_size, elapsed_s=elapsed, aggregate_mib_per_s=len(files)*file_size/MIB/elapsed if elapsed else None)
    return 0
if __name__ == "__main__": raise SystemExit(main())
