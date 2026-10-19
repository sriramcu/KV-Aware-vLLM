"""No CUDA, no PyTorch, no tokenizer or vLLM import required."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def from_path(label, rel):
    spec = importlib.util.spec_from_file_location(label, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ordering = from_path('sc_ordering', 'experiments/passage_order_policies/__init__.py')
analysis = from_path('sc_analysis', 'scripts/analyze_phase_trace.py')


class TestPolicies(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {'question': 'q1', 'sorted_passage': ['B', 'A', 'C'],
             'sorted_passage_scores': [.99, .88, .77]},
            {'question': 'q2', 'sorted_passage': ['C', 'A', 'B'],
             'sorted_passage_scores': [.5, .4, .3]},
        ]

    def test_original_is_identity(self):
        self.assertIs(ordering.apply_passage_order(self.rows, 'original'), self.rows)

    def test_canonical_is_stable_and_keeps_scores_attached(self):
        a = ordering.apply_passage_order(self.rows, 'canonical')
        b = ordering.apply_passage_order(self.rows, 'canonical')
        self.assertEqual(a, b)
        self.assertEqual(a[0]['sorted_passage'], a[1]['sorted_passage'])
        for original, result in zip(self.rows,a):
            for new_pos, rank in enumerate(result['retrieval_rank_after_order']):
                self.assertEqual(result['sorted_passage'][new_pos], original['sorted_passage'][rank])
                self.assertEqual(result['sorted_passage_scores'][new_pos], original['sorted_passage_scores'][rank])
        self.assertEqual(self.rows[0]['sorted_passage'], ['B','A','C'])

    def test_duplicates_and_bad_shape(self):
        a = ordering.apply_passage_order([{'sorted_passage':['X','X'],
             'sorted_passage_scores':[.9,.2]}], 'canonical')[0]
        self.assertEqual(a['retrieval_rank_after_order'],[0,1])
        with self.assertRaises(ValueError):
            ordering.apply_passage_order([{'sorted_passage':['A'],
                'sorted_passage_scores':[]}], 'canonical')

    def test_scheduler_snapshot_before_position_advance(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as d:
            old=os.environ.get('VLLM_PHASE_TRACE')
            old_dir=os.environ.get('VLLM_PHASE_TRACE_DIR')
            try:
                os.environ['VLLM_PHASE_TRACE']='1'
                os.environ['VLLM_PHASE_TRACE_DIR']=d
                trace=from_path('sc_trace_test','vllm/v1/metrics/phase_trace.py')
                req_a=SimpleNamespace(num_prompt_tokens=512,
                   num_computed_tokens=256,num_preemptions=0,
                   status=SimpleNamespace(name='RUNNING'))
                req_b=SimpleNamespace(num_prompt_tokens=512,
                   num_computed_tokens=512,num_preemptions=0,
                   status=SimpleNamespace(name='RUNNING'))
                scheduler=SimpleNamespace(requests={'a':req_a,'b':req_b},
                    running=[req_a,req_b])
                output=SimpleNamespace(num_scheduled_tokens={'a':256,'b':1},
                    total_num_scheduled_tokens=257)
                trace.capture_schedule(scheduler,output)
                self.assertEqual(output._sc_phase_totals,(256,1))
                req_a.num_computed_tokens += 256
                self.assertEqual(output._sc_phase_rows[0]['before'],256)
                trace.close()
            finally:
                if old is None: os.environ.pop('VLLM_PHASE_TRACE',None)
                else: os.environ['VLLM_PHASE_TRACE']=old
                if old_dir is None: os.environ.pop('VLLM_PHASE_TRACE_DIR',None)
                else: os.environ['VLLM_PHASE_TRACE_DIR']=old_dir

    def test_trace_summary_synthetic(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            record={'kind':'engine_step','ts_ns':100_000_000,
                'schedule_start_ns':0,'schedule_end_ns':1000,
                'submit_end_ns':2000,'result_end_ns':12_000_000,
                'update_end_ns':13_000_000,'prompt_compute_tokens':512,
                'decode_position_tokens':1,'waiting_remote_requests':1,
                'requests':[{'request_id':'kvaware-cold-000000',
                             'prompt_compute':512,'decode_position':0,'scheduled':512},
                            {'request_id':'kvaware-warm-000000',
                             'prompt_compute':0,'decode_position':1,'scheduled':1}]}
            import json
            (d/'phase_engine.pid1.jsonl').write_text(json.dumps(record)+'\n')
            result=analysis.run(d)
            self.assertEqual(result['steps']['mixed_steps'],1)
            self.assertEqual(result['by_phase']['cold']['prompt_compute']['sum'],512)
            self.assertEqual(result['by_phase']['warm']['decode_position']['sum'],1)


if __name__=='__main__': unittest.main()
