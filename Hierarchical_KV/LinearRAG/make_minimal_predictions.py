import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def main():
    dataset_name = "hotpotqa"
    top_k = 5
    max_questions = 200   # increase later if needed

    questions_path = Path("dataset") / dataset_name / "questions.json"
    passage_path = Path("import") / dataset_name / "passage_embedding.parquet"

    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    questions = questions[:max_questions]

    df = pd.read_parquet(passage_path)
    passage_texts = df["text"].tolist()
    passage_embeddings = np.asarray(df["embedding"].tolist(), dtype=np.float32)

    model = SentenceTransformer("model/all-mpnet-base-v2", device="cuda")

    out = []
    for item in questions:
        q = item["question"]
        q_emb = np.asarray(
            model.encode(q, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )
        scores = np.dot(passage_embeddings, q_emb)
        top_idx = np.argsort(scores)[::-1][:top_k]

        out.append({
            "question": q,
            "answer": item.get("answer", ""),
            "gold_answer": item.get("answer", ""),
            "sorted_passage": [passage_texts[i] for i in top_idx],
            "sorted_passage_scores": [float(scores[i]) for i in top_idx],
            "retrieval_mode": "minimal_dense_self_generated",
        })

    run_dir = Path("results") / dataset_name / "self_generated"
    run_dir.mkdir(parents=True, exist_ok=True)

    out_path = run_dir / "predictions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("saved ->", out_path)
    print("num predictions:", len(out))


if __name__ == "__main__":
    main()