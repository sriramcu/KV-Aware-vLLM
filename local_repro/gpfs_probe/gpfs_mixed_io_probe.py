#!/usr/bin/env python3
"""Measure one 80 MiB-file reader against 0-3 concurrent buffered writers."""
from __future__ import annotations
import argparse, errno, json, math, mmap, multiprocessing as mp, os, socket, statistics, time, traceback
from pathlib import Path
from queue import Empty
from typing import Any
MIB=1024*1024

def emit(log_path: str|Path, event: str, **fields: object) -> None:
    rec={"event":event,"wall":time.time(),"mono":time.monotonic(),"pid":os.getpid(),"host":socket.gethostname(),**fields}
    with open(log_path,"a",encoding="utf-8",buffering=1) as f: f.write(json.dumps(rec,sort_keys=True)+"\n")

def proc_io(pid: int|None=None)->dict[str,int]:
    out={}; target=pid or os.getpid()
    try:
        with open(f"/proc/{target}/io") as f:
            for line in f:
                k,v=line.split(":",1); out[k.strip()]=int(v.strip())
    except (FileNotFoundError,PermissionError,ProcessLookupError): pass
    return out

def delta(a:dict[str,int],b:dict[str,int])->dict[str,int]: return {k:a.get(k,0)-b.get(k,0) for k in a}

def meminfo()->dict[str,int]:
    wanted={"MemAvailable","Cached","Dirty","Writeback","WritebackTmp","SReclaimable"}; out={}
    with open("/proc/meminfo") as f:
        for line in f:
            k,r=line.split(":",1)
            if k in wanted: out[k]=int(r.strip().split()[0])*1024
    return out

def vmstat()->dict[str,int]:
    wanted={"nr_dirty","nr_writeback","nr_writeback_temp","pgpgin","pgpgout","pswpin","pswpout"}; out={}
    with open("/proc/vmstat") as f:
        for line in f:
            k,v=line.split()
            if k in wanted: out[k]=int(v)
    return out

def netdev()->dict[str,int]:
    rx=tx=0
    with open("/proc/net/dev") as f:
        for line in f:
            if ":" not in line: continue
            iface,rest=line.split(":",1)
            if iface.strip()=="lo": continue
            cols=rest.split(); rx+=int(cols[0]); tx+=int(cols[8])
    return {"net_rx_bytes":rx,"net_tx_bytes":tx}

def sampler(log_path:str, stop:Any, interval:float)->None:
    try:
        pv=vmstat(); pn=netdev(); pm=time.monotonic()
        while not stop.is_set():
            now=time.monotonic(); v=vmstat(); n=netdev(); e=max(now-pm,1e-9)
            emit(log_path,"system_sample",interval_s=e,**meminfo(),**v,**n,
                 pgpgin_delta=v.get("pgpgin",0)-pv.get("pgpgin",0),pgpgout_delta=v.get("pgpgout",0)-pv.get("pgpgout",0),
                 net_rx_delta=n["net_rx_bytes"]-pn["net_rx_bytes"],net_tx_delta=n["net_tx_bytes"]-pn["net_tx_bytes"])
            pv,pn,pm=v,n,now; stop.wait(interval)
    except BaseException as ex: emit(log_path,"sampler_error",error=repr(ex),traceback=traceback.format_exc())

def write_all(fd:int,payload:bytes)->int:
    mv=memoryview(payload); total=0
    while total<len(mv):
        n=os.write(fd,mv[total:])
        if n<=0: raise OSError(f"os.write returned {n}")
        total+=n
    return total

def writer(worker_id:int,phase_dir:str,file_size:int,max_files:int,start:Any,stop:Any,q:Any,log_path:str)->None:
    payload=os.urandom(file_size); files=0; total_bytes=0; times=[]
    try:
        start.wait(); emit(log_path,"writer_start",worker_id=worker_id,file_size=file_size,max_files=max_files)
        for i in range(max_files):
            if stop.is_set(): break
            path=Path(phase_dir)/f"writer_{worker_id:02d}_{i:05d}.bin"; before=proc_io(); fd=-1
            try:
                t=time.monotonic(); fd=os.open(path,os.O_CREAT|os.O_TRUNC|os.O_WRONLY,0o644); open_s=time.monotonic()-t
                t=time.monotonic(); nbytes=write_all(fd,payload); write_s=time.monotonic()-t
                t=time.monotonic(); os.close(fd); fd=-1; close_s=time.monotonic()-t
            finally:
                if fd>=0: os.close(fd)
            elapsed=open_s+write_s+close_s; d=delta(proc_io(),before); files+=1; total_bytes+=nbytes; times.append(elapsed)
            emit(log_path,"writer_file_done",worker_id=worker_id,index=i,path=str(path),bytes=nbytes,open_s=open_s,write_s=write_s,close_s=close_s,elapsed_s=elapsed,
                 application_mib_per_s=nbytes/MIB/elapsed if elapsed else None,proc_write_bytes_delta=d.get("write_bytes",0),proc_wchar_delta=d.get("wchar",0),proc_syscw_delta=d.get("syscw",0))
        s={"worker_id":worker_id,"written_files":files,"written_bytes":total_bytes,"elapsed_sum_s":sum(times),"median_file_s":statistics.median(times) if times else None,"max_file_s":max(times) if times else None}
        emit(log_path,"writer_done",**s); q.put(("writer",s))
    except BaseException as ex:
        emit(log_path,"writer_error",worker_id=worker_id,error=repr(ex),traceback=traceback.format_exc()); q.put(("writer_error",{"worker_id":worker_id,"error":repr(ex)})); raise

def direct_read(path:str,chunk:int)->tuple[int,float,float,float]:
    if not hasattr(os,"O_DIRECT"): raise OSError(errno.EOPNOTSUPP,"O_DIRECT unavailable")
    t=time.monotonic(); fd=os.open(path,os.O_RDONLY|os.O_DIRECT); open_s=time.monotonic()-t
    buf=mmap.mmap(-1,chunk); mv=memoryview(buf); total=0
    try:
        t=time.monotonic()
        while True:
            n=os.readv(fd,[mv])
            if n==0: break
            total+=n
        read_s=time.monotonic()-t
    finally:
        mv.release(); buf.close()
    t=time.monotonic(); os.close(fd); close_s=time.monotonic()-t
    return total,open_s,read_s,close_s

def buffered_read(path:str,chunk:int)->tuple[int,float,float,float]:
    t=time.monotonic(); fd=os.open(path,os.O_RDONLY); open_s=time.monotonic()-t
    try:
        if hasattr(os,"posix_fadvise") and hasattr(os,"POSIX_FADV_DONTNEED"):
            try: os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
            except OSError: pass
        total=0; t=time.monotonic()
        while True:
            b=os.read(fd,chunk)
            if not b: break
            total+=len(b)
        read_s=time.monotonic()-t
        if hasattr(os,"posix_fadvise") and hasattr(os,"POSIX_FADV_DONTNEED"):
            try: os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
            except OSError: pass
    finally:
        t=time.monotonic(); os.close(fd); close_s=time.monotonic()-t
    return total,open_s,read_s,close_s

def reader(files:list[str],chunk:int,prefer_direct:bool,start:Any,q:Any,log_path:str)->None:
    mode="direct" if prefer_direct else "buffered"; fallback=None; lats=[]; total_bytes=0; total_read=0.0
    try:
        start.wait(); emit(log_path,"reader_start",file_count=len(files),chunk_size=chunk,requested_mode=mode)
        for i,path in enumerate(files):
            before=proc_io()
            try:
                if mode=="direct": vals=direct_read(path,chunk)
                else: vals=buffered_read(path,chunk)
            except OSError as ex:
                if mode!="direct" or ex.errno not in {errno.EINVAL,errno.EOPNOTSUPP,errno.ENOTSUP}: raise
                fallback=f"{type(ex).__name__}: {ex}"; mode="buffered"; emit(log_path,"reader_direct_fallback",path=path,reason=fallback); vals=buffered_read(path,chunk)
            nbytes,open_s,read_s,close_s=vals; d=delta(proc_io(),before); elapsed=open_s+read_s+close_s
            total_bytes+=nbytes; total_read+=read_s; lats.append(read_s)
            emit(log_path,"reader_file_done",index=i,path=path,mode=mode,bytes=nbytes,open_s=open_s,read_s=read_s,close_s=close_s,elapsed_s=elapsed,
                 read_mib_per_s=nbytes/MIB/read_s if read_s else None,proc_read_bytes_delta=d.get("read_bytes",0),proc_rchar_delta=d.get("rchar",0),proc_syscr_delta=d.get("syscr",0))
        ordered=sorted(lats); p95=ordered[max(0,math.ceil(.95*len(ordered))-1)] if ordered else None
        s={"mode":mode,"direct_fallback_reason":fallback,"files":len(files),"bytes":total_bytes,"total_read_s":total_read,"aggregate_read_mib_per_s":total_bytes/MIB/total_read if total_read else None,
           "median_file_read_s":statistics.median(lats) if lats else None,"p95_file_read_s":p95,"max_file_read_s":max(lats) if lats else None}
        emit(log_path,"reader_done",**s); q.put(("reader",s))
    except BaseException as ex:
        emit(log_path,"reader_error",error=repr(ex),traceback=traceback.format_exc()); q.put(("reader_error",{"error":repr(ex)})); raise

def flush_files(phase_dir:Path,log_path:str)->dict[str,object]:
    fs=sorted(phase_dir.glob("writer_*.bin")); l=[]; t0=time.monotonic()
    for i,path in enumerate(fs):
        t=time.monotonic(); fd=os.open(path,os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
        e=time.monotonic()-t; l.append(e); emit(log_path,"phase_file_fsync_done",index=i,path=str(path),fsync_s=e)
    s={"files":len(fs),"total_fsync_s":time.monotonic()-t0,"median_fsync_s":statistics.median(l) if l else None,"max_fsync_s":max(l) if l else None}
    emit(log_path,"phase_flush_done",**s); return s

def drain(q:Any)->list[tuple[str,dict[str,object]]]:
    out=[]
    while True:
        try: out.append(q.get_nowait())
        except Empty: return out

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--root",default="/mnt/shared/gpfs/home/sriramc2/runs/gpfs_io_probe"); p.add_argument("--phases",default="0,1,2,3,0")
    p.add_argument("--reader-files-per-phase",type=int,default=25); p.add_argument("--reader-chunk-mib",type=int,default=8); p.add_argument("--writer-file-mib",type=int,default=80)
    p.add_argument("--writer-max-files",type=int,default=128); p.add_argument("--writer-headstart-s",type=float,default=3.0); p.add_argument("--sample-interval-s",type=float,default=1.0)
    p.add_argument("--buffered-reader",action="store_true"); p.add_argument("--keep-writer-files",action="store_true"); a=p.parse_args()
    root=Path(a.root); manifest_path=root/"manifest.json"
    if not manifest_path.exists(): raise FileNotFoundError(f"Missing {manifest_path}; run prepare_gpfs_probe.py first")
    m=json.loads(manifest_path.read_text()); all_files=m["files"]; phases=[int(x) for x in a.phases.split(",") if x.strip()]
    need=len(phases)*a.reader_files_per_phase
    if len(all_files)<need: raise ValueError(f"Need {need} reader files, manifest has {len(all_files)}")
    run_id=f"{socket.gethostname()}_{int(time.time())}_{os.getpid()}"; run_dir=root/"runs"/run_id; run_dir.mkdir(parents=True,exist_ok=False); main_log=run_dir/"main.jsonl"; ctx=mp.get_context("spawn")
    emit(main_log,"run_start",run_id=run_id,root=str(root),phases=phases,reader_files_per_phase=a.reader_files_per_phase,prefer_direct=not a.buffered_reader,
         slurm_job_id=os.environ.get("SLURM_JOB_ID"),cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),**meminfo())
    summaries=[]
    for pi,wcount in enumerate(phases):
        phase_dir=run_dir/f"phase_{pi:02d}_writers_{wcount}"; phase_dir.mkdir(); phase_log=phase_dir/"phase.jsonl"; rfiles=all_files[pi*a.reader_files_per_phase:(pi+1)*a.reader_files_per_phase]
        q=ctx.Queue(); wstart=ctx.Event(); rstart=ctx.Event(); wstop=ctx.Event(); sstop=ctx.Event()
        sp=ctx.Process(target=sampler,args=(str(phase_log),sstop,a.sample_interval_s),name=f"sampler-{pi}"); sp.start()
        writers=[]
        for wid in range(wcount):
            pr=ctx.Process(target=writer,args=(wid,str(phase_dir),a.writer_file_mib*MIB,a.writer_max_files,wstart,wstop,q,str(phase_log)),name=f"writer-{pi}-{wid}"); pr.start(); writers.append(pr)
        rp=ctx.Process(target=reader,args=(rfiles,a.reader_chunk_mib*MIB,not a.buffered_reader,rstart,q,str(phase_log)),name=f"reader-{pi}"); rp.start()
        emit(phase_log,"phase_start",phase_index=pi,writer_count=wcount,reader_files=rfiles,writer_headstart_s=a.writer_headstart_s); wstart.set()
        if wcount: time.sleep(a.writer_headstart_s)
        rstart.set(); rp.join(); wstop.set()
        for pr in writers: pr.join()
        flush=flush_files(phase_dir,str(phase_log)); results=drain(q); sstop.set(); sp.join(timeout=5)
        if sp.is_alive(): sp.terminate(); sp.join()
        summary={"phase_index":pi,"writer_count":wcount,"reader_exitcode":rp.exitcode,"writer_exitcodes":[p.exitcode for p in writers],"flush":flush,"results":results}; summaries.append(summary)
        emit(phase_log,"phase_done",**summary); emit(main_log,"phase_done",**summary)
        if not a.keep_writer_files:
            for path in phase_dir.glob("writer_*.bin"): path.unlink()
            emit(phase_log,"phase_writer_files_deleted")
        if rp.exitcode!=0 or any(p.exitcode!=0 for p in writers): emit(main_log,"run_abort_due_to_child_error",phase=pi); return 2
    spath=run_dir/"summary.json"; spath.write_text(json.dumps(summaries,indent=2)+"\n"); emit(main_log,"run_done",summary=str(spath),run_dir=str(run_dir)); print(f"GPFS probe completed: {run_dir}\nSummary: {spath}",flush=True); return 0
if __name__=="__main__": raise SystemExit(main())
