"""
Week 2: Generate Sentence-BERT embeddings for repository text (description + topics + README).
Uses shared build_embedding_text so query and index are comparable. Loads data/repos.json, embeds, saves for FAISS.

Usage:
  python -m src.embeddings
  python -m src.embeddings --repos data/repos.json --out-dir data
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from src.embedding_text import build_embedding_text
except ImportError:
    from embedding_text import build_embedding_text

def main():
    parser = argparse.ArgumentParser(description="Embed READMEs with Sentence-BERT.")
    parser.add_argument("--repos", type=str, default="data/repos.json", help="Path to repos.json")
    parser.add_argument("--out-dir", type=str, default="data", help="Directory for embeddings.npy and repo_index.json")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="Sentence-BERT model name")
    parser.add_argument("--readme-chars", type=int, default=3000, help="Max README chars to embed (default 3000)")
    args = parser.parse_args()

    repos_path = Path(args.repos)
    if not repos_path.exists():
        raise FileNotFoundError(f"Not found: {repos_path}")

    with open(repos_path, encoding="utf-8") as f:
        repos = json.load(f)

    texts = [
        build_embedding_text(r, readme_max_chars=args.readme_chars)
        for r in repos
    ]
    repo_names = [r["repo"] for r in repos]

    print(f"Loading model {args.model}...")
    model = SentenceTransformer(args.model)
    print(f"Embedding {len(texts)} READMEs...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", embeddings)
    with open(out_dir / "repo_index.json", "w", encoding="utf-8") as f:
        json.dump(repo_names, f, indent=0)

    print(f"Saved embeddings shape {embeddings.shape} and repo_index.json to {out_dir}")

if __name__ == "__main__":
    main()
