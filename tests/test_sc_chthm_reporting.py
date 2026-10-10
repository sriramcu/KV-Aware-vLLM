"""Standalone regression tests for offline CHTHM reporting (no GPU dependencies)."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

FILE = Path(__file__).resolve().parents[1] / 'scripts/analyze_mp_congestion_chthm.py'
spec = importlib.util.spec_from_file_location('analyze_mp_congestion_chthm', FILE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class ChthmReportingTests(unittest.TestCase):
    def test_phase_suffix_preserves_full_request_id(self):
        rid = 'cmpl-kvaware-warm-000572-0-98cfbb44'
        self.assertEqual(mod.base_request_id(rid), 'cmpl-kvaware-warm-000572')
        self.assertEqual(mod.request_phase(rid, {'cmpl-kvaware-warm-000572':'warm'}), 'warm')
        self.assertEqual(mod.request_phase('cmpl-kvaware-cold-000000-0-ba3f55a1', {}), 'cold')
        self.assertEqual(mod.request_phase('unrecognized', {}), 'unknown')

    def test_phase_map_deduplicates_warm_order_files(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)
            (p/'cold_results.jsonl').write_text(json.dumps({'request_id':'cmpl-kvaware-cold-000000','phase':'cold'})+'\n')
            row=json.dumps({'request_id':'cmpl-kvaware-warm-000000','phase':'warm'})+'\n'
            (p/'warm_results_execution_order.jsonl').write_text(row)
            (p/'warm_results_source_order.jsonl').write_text(row)
            self.assertEqual(mod.phase_map(p), {'cmpl-kvaware-cold-000000':'cold','cmpl-kvaware-warm-000000':'warm'})

    def test_validate_realistic_raw_source_map(self):
        rec={'chunk_size':512, 'requested_chunks':5, 'source_map':'CCDMM',
             'l1_hit_chunks':2, 'l2_hit_chunks':1,'miss_chunks':2,'reachable_prefix_chunks':3}
        mod.validate_raw_geometry('req',rec)
        self.assertEqual(mod.source_token_counts({'chunk_size':512, 'start_token':512, 'source_tiers':['L1','L2','MISS']},
                                                  1024,2048), {'L1':0,'L2':512,'MISS':512})
        rec['miss_chunks']=1
        with self.assertRaises(ValueError):mod.validate_raw_geometry('req',rec)

    def test_partial_source_map_coverage(self):
        # External opportunity [1536,4096); only [2048,3584) was measured.
        self.assertEqual(mod.overlap(1536,4096,2048,3584),1536)
        self.assertEqual(4096-1536-1536,1024)

if __name__=='__main__': unittest.main()
