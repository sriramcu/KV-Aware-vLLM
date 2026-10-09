#!/usr/bin/env bash
set -euo pipefail
# Run from the KV-Aware-vLLM repository root after applying kvaware_refactor_core.patch.
# This only removes files that are absent from the final refactored tree.
rm -f -- Hierarchical_KV/shortq_placement/tier_mapping.py
rm -f -- gnn_aware_vpc_l1_backing.patch
rm -f -- gnn_dynamic_prompt_only_fix.patch
rm -f -- kv_recompute_fa_version_diagnostic.patch
rm -f -- kvaware_20541_recovery_fix.patch
rm -f -- kvaware_20541_recovery_fix_v2_no_connector.patch
rm -f -- kvaware_20588_followup_v3.patch
rm -f -- kvaware_dynamic_vpc_three_fixes.patch
rm -f -- kvaware_gnn_batching_bundle.zip
rm -f -- kvaware_gnn_l2_backing_store_barrier.patch
rm -f -- kvaware_l0_smoke_force_external_retrieve.patch
rm -f -- kvaware_l0_smoke_store.patch
rm -f -- kvaware_l0_v1_infrastructure.patch
rm -f -- kvaware_l0_vpc_imitation_200g_pf2.patch
rm -f -- kvaware_lookup_pf_separation_v1.patch
rm -f -- kvaware_lookup_pf_separation_v2.patch
rm -f -- kvaware_lookup_rpc_l1snapshot_v1.patch
rm -f -- kvaware_mutual_prefix_grid.patch
rm -f -- kvaware_mutual_prefix_stale_bridge_runfolder_fix.patch
rm -f -- kvaware_prefix_hole_diagnostics.patch
rm -f -- kvaware_shortq_gnn_exclusive_v1.patch
rm -f -- kvaware_shortq_gnn_inference_hotfix.patch
rm -f -- kvaware_shortq_gnn_prompt_only_store_fix.patch
rm -f -- kvaware_stage1_freshness_mp_diag.patch
rm -f -- kvaware_storecap_put_bytes_v1.patch
rm -f -- kvaware_warm_random_order_v1.patch
rm -f -- scripts/0001-shortq-inference-batching.patch
rm -f -- scripts/run_h100_l0_smoke.sbatch
rm -f -- scripts/run_h100_l0_vpc_imitation_q650_200g_pf2_cleanall.sbatch
rm -f -- scripts/run_h100_l0off_q650_l1_inflight_grid_cleanall.sbatch
rm -f -- scripts/run_h100_shortq_gnn_q650.sbatch
rm -f -- scripts/run_h100_shortq_gnn_smoke.sbatch
rm -f -- scripts/run_l40_compare_nognn_vs_gnn.sbatch
rm -f -- scripts/run_l40_shortq_gnn_q200.sbatch
rm -f -- scripts/run_l40_shortq_gnn_q200_cpu_top2.sbatch
rm -f -- scripts/run_l40_shortq_gnn_q200_gpu6_cpu5.sbatch
rm -f -- scripts/run_l40_shortq_gnn_smoke.sbatch
rm -f -- scripts/run_shortq_gnn_dynamic_vpc_common.sh.before_shared_worker_edit
rm -f -- scripts/run_shortq_gnn_exclusive_common.sh
rm -f -- third_party/LMCache/lmcache/v1/distributed/l0_manager.py
rm -f -- third_party/LMCache/tests/v1/distributed/test_l0_manager.py
rm -f -- third_party/LMCache/tests/v1/distributed/test_l0_union_lookup.py
