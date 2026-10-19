import unittest, tempfile, json
from pathlib import Path
from collections import OrderedDict, Counter
import analyze_medical_reuse as m

class ToyTokenizer:
    is_fast=True
    def apply_chat_template(self,messages,tokenize,add_generation_prompt):
        return '<|s|>'+messages[0]['content']+'<|u|>'+messages[1]['content']+'<|a|>'
    def __call__(self,text,add_special_tokens,return_offsets_mapping,truncation):
        result={'input_ids':[ord(x) for x in text]}
        if return_offsets_mapping:result['offset_mapping']=[(i,i+1) for i in range(len(text))]
        return result

class Tests(unittest.TestCase):
    def setUp(self):
        self.encoder=m.PromptEncoder.__new__(m.PromptEncoder)
        self.encoder.tokenizer=ToyTokenizer();self.encoder.size=512
        self.encoder.enc_cache=OrderedDict();self.encoder.max_cache=1000
        self.encoder.key_cache=OrderedDict();self.encoder.max_key_cache=1000
        self.encoder.render_count=0;self.encoder.encode_time=0
        a='X'*1000;b='Y'*1000;c='Z'*1000
        records=[{'question':'Q1','sorted_passage':[a,b,c]},
                 {'question':'Q2','sorted_passage':[b,a,c]},
                 {'question':'Q3','sorted_passage':[a,c,b]},
                 {'question':'Q4','sorted_passage':[c,b,a]}]
        self.recs=m.normalize_records(records,5,None)
        self.pop=Counter(p for r in self.recs for p in r['ids'])

    def test_complete_prefix_hash(self):
        a=self.recs[0];b=self.recs[1]
        aa=self.encoder.encode(a['passages'],a['question'])
        bb=self.encoder.encode(b['passages'],b['question'])
        self.assertNotEqual(aa['keys'][0],bb['keys'][0])
        align=self.encoder.encode(b['passages'][1:2]+b['passages'][:1]+b['passages'][2:],b['question'])
        self.assertEqual(aa['keys'][0],align['keys'][0])
        self.assertNotEqual(aa['tokens'],align['tokens'])  # different question suffix can remain a partial chunk

    def test_planner_improves_prefix(self):
        def test(strategy):
            return m.scenario(self.recs,list(range(4)),self.encoder,strategy,None,2,'same',1,0,self.pop)[0]['cold']['useful_prefix_hit_chunks']
        self.assertGreater(test('greedy'),test('original'))

    def test_boundary(self):
        item=self.encoder.boundaries(self.recs[0]['passages'],self.recs[0]['question'])
        self.assertGreaterEqual(item['mixed_passage_full_chunk_count'],0)
        self.assertEqual(item['source_passage_count'],3)

    def test_passage(self):
        with tempfile.TemporaryDirectory() as d:
            p,rr=m.passage_analysis(self.recs,list(range(4)),Path(d),100,0)
            self.assertEqual(p['distinct_passages'],3)
            self.assertEqual(rr[1]['best_shared_passage_set_with_prior'],3)
            self.assertTrue((Path(d)/'pairwise_overlap.csv').exists())

    def test_hindsight(self):
        stat,rows=m.hindsight_all_original(self.recs,list(range(4)),self.encoder)
        self.assertEqual(stat['requests'],4)
        self.assertTrue(all(r['additional_hits']>=0 for r in rows))

    def test_sweep(self):
        rows=m.chunk_size_sweep(self.recs,list(range(4)),self.encoder,[128,512,1024])
        self.assertEqual(len(rows),3)
        self.assertGreater(rows[0]['full_chunk_occurrences'],rows[-1]['full_chunk_occurrences'])

if __name__=='__main__':unittest.main()
