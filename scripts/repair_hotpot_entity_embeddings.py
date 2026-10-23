#!/usr/bin/env python3
"""Regenerate ONLY HotpotQA's missing entity_embedding.parquet from existing GraphML.

Expected input graph was produced by Hierarchical_KV/LinearRAG/src/LinearRAG.py,
whose entity vertices contain name='entity-'+md5(content) and content=text.
Uses the exact local SentenceTransformer model used for the passage embeddings.
Does not rerun spaCy/NER, graph construction, or write to scratch.
"""
import argparse
import hashlib
import os
from pathlib import Path
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, default=Path.cwd())
    ap.add_argument('--dataset', default='hotpotqa')
    ap.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    repo = a.repo.resolve()
    base = repo / 'Hierarchical_KV' / 'LinearRAG'
    imp = base / 'import' / a.dataset
    graph_file = imp / 'LinearRAG.graphml'
    passage_file = imp / 'passage_embedding.parquet'
    target = imp / 'entity_embedding.parquet'
    model_dir = base / 'model' / 'all-mpnet-base-v2'
    if target.exists():
        raise SystemExit(f'Refusing to overwrite existing file: {target}')
    for p in (graph_file, passage_file, model_dir):
        if not p.exists():
            raise SystemExit(f'Missing required input: {p}')

    import igraph as ig
    import pandas as pd

    graph = ig.Graph.Read_GraphML(str(graph_file))
    names = graph.vs['name']
    if 'content' not in graph.vs.attributes():
        raise SystemExit('GraphML lacks vertex content attributes; cannot recover entity texts safely.')
    contents = graph.vs['content']
    entries = []
    for name, content in zip(names, contents):
        name = str(name)
        if not name.startswith('entity-'):
            continue
        if not isinstance(content, str) or not content:
            raise SystemExit(f'Entity vertex lacks usable text: {name}')
        expected = 'entity-' + hashlib.md5(content.encode()).hexdigest()
        if expected != name:
            raise SystemExit(f'Entity text/hash mismatch: {name}; refusing to generate incorrect embeddings')
        entries.append((name, content))
    if not entries:
        raise SystemExit('No entity-* vertices found in graph; refusing to write empty artifact')
    if len({n for n, _ in entries}) != len(entries):
        raise SystemExit('Duplicate entity hashes in graph')
    entries.sort(key=lambda x: x[0])

    passage_df = pd.read_parquet(passage_file, columns=['embedding'])
    if passage_df.empty:
        raise SystemExit('Passage embeddings are empty')
    dim = len(passage_df['embedding'].iloc[0])
    del passage_df
    print(f'Graph: {graph.vcount()} vertices, {graph.ecount()} edges; '
          f'{len(entries)} entity nodes, passage dim={dim}', flush=True)
    if a.dry_run:
        print('Dry run complete; no file written.', flush=True)
        return

    from sentence_transformers import SentenceTransformer
    import numpy as np
    model = SentenceTransformer(str(model_dir), device=a.device)
    if int(model.get_sentence_embedding_dimension()) != dim:
        raise SystemExit('Embedding model dimension does not match existing passage embeddings')
    texts = [t for _, t in entries]
    embeddings = model.encode(texts, normalize_embeddings=True,
                              show_progress_bar=True, batch_size=a.batch_size,
                              convert_to_numpy=True)
    if embeddings.shape != (len(entries), dim) or not np.isfinite(embeddings).all():
        raise SystemExit('Bad embedding dimensions or nonfinite values; not writing output')
    df = pd.DataFrame({'hash_id': [h for h, _ in entries],
                       'text': texts,
                       'embedding': list(embeddings.astype(np.float32))})
    tmp = target.with_name(target.name + f'.partial.{os.getpid()}')
    try:
        df.to_parquet(tmp, index=False)
        verification = pd.read_parquet(tmp, columns=['hash_id', 'text', 'embedding'])
        if len(verification) != len(entries):
            raise RuntimeError('Written parquet record count mismatch')
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()
    print(f'CREATED: {target} ({len(entries)} entity embeddings, {dim} dimensions)', flush=True)


if __name__ == '__main__':
    main()
