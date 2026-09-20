import argparse
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from src.config import LinearRAGConfig
from src.LinearRAG import LinearRAG


def load_passages(dataset_name: str):
    chunks_path = Path("dataset") / dataset_name / "chunks.json"
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return [f"{idx}:{chunk}" for idx, chunk in enumerate(chunks)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--embedding_model_path", default="model/all-mpnet-base-v2")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    embedding_model_path = args.embedding_model_path

    print("Loading embedding model:", embedding_model_path, flush=True)
    embedding_model = SentenceTransformer(embedding_model_path, device="cuda")

    print("Loading passages for:", dataset_name, flush=True)
    passages = load_passages(dataset_name)
    print("num passages:", len(passages), flush=True)

    config = LinearRAGConfig(
        dataset_name=dataset_name,
        embedding_model=embedding_model,
        spacy_model="en_core_web_trf",
        working_dir="./import",
        batch_size=128,
        max_workers=8,
        retrieval_top_k=5,
        llm_model=None,
    )

    print("Building LinearRAG import artifacts...", flush=True)
    rag = LinearRAG(config)
    rag.index(passages)

    print(f"Done. Expected outputs under import/{dataset_name}/", flush=True)


if __name__ == "__main__":
    main()
    