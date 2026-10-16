L40 GNN MNS Replication — corrected NFS cleaning launcher (v2)
===================================================================

Corrected after user feedback: L40 has no /scratch2. Its LMCache L2 is on
NFS at /mnt/shared/gpfs/home/sriramc2/runs/kvaware/lmcache_mp_l2_l40.

Changes compared with the original L40 bundle:
 - CLEAN_OLD_LMCACHE=1 (REQUIRED, hard-set in the runner).
 - The job-local common script's clean_old_lmcache() chooses the NFS
   L40 cache root when PROFILE=l40_q650; the H100 /scratch2 branch is
   preserved only for other profiles.
 - BEFORE EACH run, delete all entries beneath precisely the L40
   lmcache_mp_l2_l40 NFS cache root. Exact-path and non-symlink guards
   protect against accidentally cleaning elsewhere.
 - AFTER EACH run, remove just its job-specific L2 directory to avoid
   an orphaned final-run cache consuming the NFS quota. This is a
   best-effort cleanup; if Slurm kills the job abruptly, leftovers
   may require manual cleaning before any later runs.
 - Cold + warm run data (results, metrics, logs, placement and VPC stats)
   remain under runs/kvaware/l40_shortq_gnn_dynamic_vpc_q650_* and are
   NOT part of the L2 cleanup.

WARNING: This intentionally deletes old LMCache L2 cache objects in the
existing L40 NFS root. Do not submit while another run depends on files
under that root. It does NOT delete other NFS directories, your code,
H100 scratch data, or historical experiment-result directories.

Runs (exactly four sbatch commands, afterany chain):
  MNS10 threshold8, MNS7 threshold8, MNS4 threshold8, MNS10 threshold22.
All are true-DROP GNN, selective VPC, Llama-3.1-8B, TP1, Q650 650+650.

INSTALL (from repository root):
  unzip -o /path/to/l40_gnn_mns_q650_nfs_cleanup_v2_bundle.zip -d .
  bash -n scripts/run_l40_gnn_mns_replication_q650_nfs_clean.sbatch
  bash -n scripts/submit_l40_gnn_mns_replication_q650_nfs_clean.sh

SUBMIT:
  bash scripts/submit_l40_gnn_mns_replication_q650_nfs_clean.sh

The wrapper refuses to launch if the live common runner no longer matches
expected v22/v3 source snippets. A Slurm job that fails early still lets
later grid jobs start because the chain uses afterany.
