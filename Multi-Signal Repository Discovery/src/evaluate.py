"""
Evaluation framework for Multi-Signal Repository Discovery.

Measures: Precision@K, MRR, NDCG@K
Modes:
  - full:       all four signals combined (your system)
  - text:       SBERT text similarity only
  - deps:       dependency overlap only
  - topics:     topic/keyword overlap only
  - tfidf:      TF-IDF baseline (bag-of-words on README)
  - topic_tag:  exact GitHub topic tag overlap only (simplest baseline)

Usage:
  # Run full evaluation across all variants
  python -m src.evaluate --data-dir pt2/datasets_expanded \
      --repos pt2/datasets_expanded/repos_expanded.json

  # Run a single variant
  python -m src.evaluate --data-dir pt2/datasets_expanded \
      --repos pt2/datasets_expanded/repos_expanded.json \
      --variant text

  # Save results to JSON
  python -m src.evaluate --data-dir pt2/datasets_expanded \
      --repos pt2/datasets_expanded/repos_expanded.json \
      --out results/eval_results.json
"""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

GROUND_TRUTH = {

    "scikit-learn/scikit-learn": {
        "yzhao062/pyod":                        2,
        "cleanlab/cleanlab":                    2,
        "Data-Centric-AI-Community/ydata-profiling": 1,
        "pandas-dev/pandas":                    1,
        "facebookresearch/fairseq":             1,
        "donnemartin/data-science-ipython-notebooks": 1,
    },
    "karpathy/micrograd": {
        "tinygrad/tinygrad":                    2,
        "keras-team/keras":                     1,
        "pytorch/pytorch":                      2,
        "google-deepmind/sonnet":               1,
        "yunjey/pytorch-tutorial":              1,
    },
    "huggingface/transformers": {
        "huggingface/diffusers":                2,
        "huggingface/accelerate":               2,
        "huggingface/peft":                     2,
        "speechbrain/speechbrain":              2,
        "espnet/espnet":                        1,
        "mlfoundations/open_clip":              1,
        "facebookresearch/fairseq":             2,
        "jadore801120/attention-is-all-you-need-pytorch": 2,
    },

    "pallets/flask": {
        "bottlepy/bottle":              2,
        "vitalik/django-ninja":         1,
        "aio-libs/aiohttp":             1,
        "fastapi/fastapi":              2,
    },
    "tiangolo/fastapi": {
        "fastapi/fastapi":              2,
        "fastapi/sqlmodel":             2,
        "fastapi/typer":                2,
        "vitalik/django-ninja":         2,
        "aio-libs/aiohttp":             1,
        "openai/openai-python":         1,
    },

    "expressjs/express": {
        "nodejs/node-gyp":              1,
        "frappe/frappe":                1,
    },
    "lodash/lodash": {
        "plotly/dash":                  1,
        "pyscript/pyscript":            1,
    },

    "TheAlgorithms/Python": {
        "keon/algorithms":              2,
        "donnemartin/interactive-coding-challenges": 1,
        "Jack-Cherish/Machine-Learning": 1,
    },
    "trekhleb/javascript-algorithms": {
        "trekhleb/learn-python":        2,
        "Jack-Cherish/Machine-Learning": 1,
        "donnemartin/data-science-ipython-notebooks": 1,
    },

    "antirez/redis": {
        "redis/redis-py":               2,
        "rq/rq":                        2,
        "getredash/redash":             1,
    },

    "openai/gym": {
        "thu-ml/tianshou":              2,
        "DLR-RM/stable-baselines3":     2,
        "openai/baselines":             2,
        "google-deepmind/dm_control":   2,
        "MorvanZhou/Reinforcement-learning-with-tensorflow": 1,
        "hill-a/stable-baselines":      2,
    },

    "psf/black": {
        "PyCQA/flake8":                 2,
        "PyCQA/isort":                  2,
        "google/yapf":                  2,
        "astral-sh/ruff":               2,
        "mypy-lang/mypy":               1,
        "pre-commit/pre-commit":        1,
    },

    "matplotlib/matplotlib": {
        "mwaskom/seaborn":                       2,
        "vega/altair":                           2,
        "plotly/dash":                           2,
        "numpy/numpy":                           1,
        "rougier/scientific-visualization-book": 1,
    },

    "huggingface/diffusers": {
        "huggingface/transformers":              2,
        "huggingface/peft":                      2,
        "Tencent-Hunyuan/HunyuanVideo":          1,
        "mlfoundations/open_clip":               1,
    },

    "ggerganov/whisper.cpp": {
        "SYSTRAN/faster-whisper":                2,
        "openai/whisper":                        2,
        "chidiwilliams/buzz":                    2,
        "coqui-ai/TTS":                          1,
        "espnet/espnet":                         1,
    },

    "sqlalchemy/sqlalchemy": {
        "fastapi/sqlmodel":                      2,
        "coleifer/peewee":                       2,
        "harelba/q":                             1,
    },

    "szagoruyko/attention-transfer": {
        "Cadene/pretrained-models.pytorch":      2,
        "jadore801120/attention-is-all-you-need-pytorch": 2,
        "mlfoundations/open_clip":               1,
        "eriklindernoren/PyTorch-GAN":           1,
    },
}

def _relevant_set(judgments: dict, min_rel: int = 1) -> set:
    return {repo for repo, score in judgments.items() if score >= min_rel}

def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not retrieved or not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for r in top_k if r in relevant)
    return hits / k

def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, r in enumerate(retrieved, 1):
        if r in relevant:
            return 1.0 / i
    return 0.0

def dcg_at_k(retrieved: list[str], judgments: dict, k: int) -> float:
    score = 0.0
    for i, r in enumerate(retrieved[:k], 1):
        rel = judgments.get(r, 0)
        score += rel / math.log2(i + 1)
    return score

def idcg_at_k(judgments: dict, k: int) -> float:
    sorted_rels = sorted(judgments.values(), reverse=True)[:k]
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(sorted_rels))

def ndcg_at_k(retrieved: list[str], judgments: dict, k: int) -> float:
    idcg = idcg_at_k(judgments, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(retrieved, judgments, k) / idcg

def _norm_deps(deps) -> set:
    if not deps:
        return set()
    return {str(x).strip().lower() for x in deps if x}

def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "this", "that", "these", "those", "it", "its",
    "as", "up", "if", "not", "no", "so", "we", "you", "i", "my", "your",
    "our", "their", "can", "all", "more", "also", "than", "then", "when",
    "how", "what", "which", "who", "using", "used", "use", "uses", "just",
    "into", "like", "get", "set", "new", "one", "two", "any", "each",
    "about", "after", "before", "between", "through", "project", "repo",
    "repository", "github", "code", "example", "examples", "based",
    "support", "file", "files", "add", "run", "make", "build", "install",
}

def _extract_keywords(text: str, n: int = 15) -> set:
    words = re.findall(r'\b[a-z][a-z0-9_-]{2,}\b', text.lower())
    filtered = [w for w in words if w not in _STOPWORDS]
    counts = Counter(filtered)
    return {w for w, _ in counts.most_common(n)}

def _get_topics(repo_dict: dict) -> set:
    topics = set(repo_dict.get("topics") or [])
    if topics:
        return topics
    desc = repo_dict.get("description") or ""
    readme = (repo_dict.get("readme_text") or "")[:2000]
    return _extract_keywords(desc + " " + readme, n=15)

def _tfidf_scores(query_repo: dict, candidates: list[dict]) -> list[tuple[str, float]]:
    """TF-IDF cosine similarity baseline using README + description."""
    from collections import defaultdict

    def tokenize(text):
        return re.findall(r'\b[a-z][a-z0-9]{1,}\b', (text or "").lower())

    def build_tf(tokens):
        counts = Counter(tokens)
        total = max(len(tokens), 1)
        return {w: c / total for w, c in counts.items()}

    all_docs = [query_repo] + candidates
    doc_freq = defaultdict(int)
    tokenized = []
    for doc in all_docs:
        text = (doc.get("description") or "") + " " + (doc.get("readme_text") or "")[:3000]
        tokens = set(tokenize(text))
        tokenized.append(tokenize(text))
        for t in tokens:
            doc_freq[t] += 1

    n_docs = len(all_docs)

    def tfidf_vec(tokens):
        tf = build_tf(tokens)
        return {w: tf_val * math.log(n_docs / (doc_freq[w] + 1))
                for w, tf_val in tf.items() if w in doc_freq}

    def cosine(v1, v2):
        common = set(v1) & set(v2)
        if not common:
            return 0.0
        dot = sum(v1[w] * v2[w] for w in common)
        norm1 = math.sqrt(sum(x ** 2 for x in v1.values()))
        norm2 = math.sqrt(sum(x ** 2 for x in v2.values()))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    q_vec = tfidf_vec(tokenized[0])
    results = []
    for i, cand in enumerate(candidates):
        c_vec = tfidf_vec(tokenized[i + 1])
        score = cosine(q_vec, c_vec)
        results.append((cand["repo"], score))
    return sorted(results, key=lambda x: x[1], reverse=True)

def _topic_tag_scores(query_repo: dict, candidates: list[dict]) -> list[tuple[str, float]]:
    """Baseline: exact GitHub topic tag overlap only (no keyword fallback)."""
    q_topics = set(query_repo.get("topics") or [])
    results = []
    for cand in candidates:
        c_topics = set(cand.get("topics") or [])
        score = _jaccard(q_topics, c_topics) if (q_topics or c_topics) else 0.0
        results.append((cand["repo"], score))
    return sorted(results, key=lambda x: x[1], reverse=True)

def _text_only_scores(query_embed, embeddings, repo_index, repos_map, k) -> list[tuple[str, float]]:
    import faiss
    q = query_embed.copy()
    faiss.normalize_L2(q)
    idx_local = faiss.IndexFlatIP(q.shape[1])
    emb = embeddings.copy().astype(np.float32)
    faiss.normalize_L2(emb)
    idx_local.add(emb)
    scores, indices = idx_local.search(q, k + 1)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        name = repo_index[idx]
        results.append((name, float(score)))
    return results

def _deps_only_scores(query_repo: dict, candidates: list[dict]) -> list[tuple[str, float]]:
    q_deps = _norm_deps(query_repo.get("dependencies"))
    results = []
    for cand in candidates:
        c_deps = _norm_deps(cand.get("dependencies"))
        score = _jaccard(q_deps, c_deps)
        results.append((cand["repo"], score))
    return sorted(results, key=lambda x: x[1], reverse=True)

def _topics_only_scores(query_repo: dict, candidates: list[dict]) -> list[tuple[str, float]]:
    q_topics = _get_topics(query_repo)
    results = []
    for cand in candidates:
        c_topics = _get_topics(cand)
        score = _jaccard(q_topics, c_topics)
        results.append((cand["repo"], score))
    return sorted(results, key=lambda x: x[1], reverse=True)

def evaluate(
    data_dir: Path,
    repos_path: Optional[Path],
    k_values: list[int],
    variants: list[str],
    verbose: bool = True,
) -> dict:
    import faiss

    data_dir = Path(data_dir)
    repos_file = repos_path or (data_dir / "repos.json")

    with open(repos_file, encoding="utf-8") as f:
        repos_list = json.load(f)

    embeddings = np.load(data_dir / "embeddings.npy").astype(np.float32)
    with open(data_dir / "repo_index.json", encoding="utf-8") as f:
        repo_index = json.load(f)

    repos_map = {r["repo"]: r for r in repos_list}
    repo_to_embed_idx = {name: i for i, name in enumerate(repo_index)}

    eval_queries = {
        q: judgments
        for q, judgments in GROUND_TRUTH.items()
        if q in repo_to_embed_idx
    }

    if verbose:
        print(f"\nEvaluating {len(eval_queries)} queries "
              f"(skipped {len(GROUND_TRUTH) - len(eval_queries)} not in index)")
        print(f"Variants: {variants}")
        print(f"K values: {k_values}\n")

    faiss_index = None
    if "full" in variants:
        faiss_index = faiss.read_index(str(data_dir / "faiss.index"))

    all_results = {}

    for variant in variants:
        if verbose:
            print(f"{'='*60}")
            print(f"Variant: {variant.upper()}")
            print(f"{'='*60}")

        query_metrics = []

        for query_repo, judgments in eval_queries.items():
            q_idx = repo_to_embed_idx[query_repo]
            q_embed = embeddings[q_idx: q_idx + 1].copy()
            q_dict = repos_map.get(query_repo, {"repo": query_repo})
            relevant = _relevant_set(judgments, min_rel=1)

            if variant == "full":
                try:
                    from src.similarity import get_similar
                except ImportError:
                    from similarity import get_similar
                results, err = get_similar(
                    query_repo, max(k_values) + 5, data_dir,
                    index=faiss_index,
                    repos=repos_list,
                    repo_index=repo_index,
                    embeddings=embeddings,
                    repos_path=repos_path,
                )
                if err or not results:
                    continue
                ranked = [r["repo"] for r in results]

            elif variant == "text":
                scored = _text_only_scores(q_embed, embeddings, repo_index, repos_map, max(k_values) + 5)
                ranked = [name for name, _ in scored if name != query_repo]

            elif variant == "deps":
                candidates = [r for r in repos_list if r["repo"] != query_repo]
                scored = _deps_only_scores(q_dict, candidates)
                ranked = [name for name, _ in scored]

            elif variant == "topics":
                candidates = [r for r in repos_list if r["repo"] != query_repo]
                scored = _topics_only_scores(q_dict, candidates)
                ranked = [name for name, _ in scored]

            elif variant == "tfidf":
                candidates = [r for r in repos_list if r["repo"] != query_repo]
                scored = _tfidf_scores(q_dict, candidates)
                ranked = [name for name, _ in scored]

            elif variant == "topic_tag":
                candidates = [r for r in repos_list if r["repo"] != query_repo]
                scored = _topic_tag_scores(q_dict, candidates)
                ranked = [name for name, _ in scored]

            else:
                continue

            mrr = reciprocal_rank(ranked, relevant)
            metrics_for_query = {"query": query_repo, "mrr": mrr}
            for k in k_values:
                p_k = precision_at_k(ranked, relevant, k)
                n_k = ndcg_at_k(ranked, judgments, k)
                metrics_for_query[f"p@{k}"] = p_k
                metrics_for_query[f"ndcg@{k}"] = n_k

            query_metrics.append(metrics_for_query)

            if verbose:
                hits = [r for r in ranked[:max(k_values)] if r in relevant]
                print(f"\n  Query: {query_repo}")
                print(f"  Relevant (ground truth): {len(relevant)} repos")
                print(f"  MRR: {mrr:.3f}  |  " +
                      "  ".join(f"P@{k}={metrics_for_query[f'p@{k}']:.3f}" for k in k_values) +
                      "  |  " +
                      "  ".join(f"NDCG@{k}={metrics_for_query[f'ndcg@{k}']:.3f}" for k in k_values))
                if hits:
                    print(f"  Hits in top-{max(k_values)}: {hits}")
                else:
                    print(f"  No hits in top-{max(k_values)} "
                          f"(relevant repos may not be in the 1000-repo index)")

        if not query_metrics:
            continue

        agg = {"variant": variant, "n_queries": len(query_metrics)}
        agg["mean_mrr"] = sum(m["mrr"] for m in query_metrics) / len(query_metrics)
        for k in k_values:
            agg[f"mean_p@{k}"] = sum(m[f"p@{k}"] for m in query_metrics) / len(query_metrics)
            agg[f"mean_ndcg@{k}"] = sum(m[f"ndcg@{k}"] for m in query_metrics) / len(query_metrics)
        agg["per_query"] = query_metrics
        all_results[variant] = agg

        if verbose:
            print(f"\n  ── Aggregate ({variant}) ──")
            print(f"  Mean MRR:     {agg['mean_mrr']:.3f}")
            for k in k_values:
                print(f"  Mean P@{k}:    {agg[f'mean_p@{k}']:.3f}")
                print(f"  Mean NDCG@{k}: {agg[f'mean_ndcg@{k}']:.3f}")

    if verbose and len(all_results) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY TABLE")
        print(f"{'='*60}")
        header = f"{'Variant':<12}" + "  MRR  " + "".join(
            f" P@{k}  NDCG@{k}" for k in k_values
        )
        print(header)
        print("-" * len(header))
        for variant, res in all_results.items():
            row = f"{variant:<12}  {res['mean_mrr']:.3f}  "
            row += "  ".join(
                f"{res[f'mean_p@{k}']:.3f}   {res[f'mean_ndcg@{k}']:.3f}"
                for k in k_values
            )
            print(row)

    return all_results

def main():
    parser = argparse.ArgumentParser(description="Evaluate the multi-signal repo discovery system.")
    parser.add_argument("--data-dir", type=str, default="pt2/datasets_expanded",
                        help="Directory with embeddings.npy, faiss.index, repo_index.json")
    parser.add_argument("--repos", type=str, default=None,
                        help="Path to repos JSON (default: <data-dir>/repos.json)")
    parser.add_argument("--k", type=str, default="5,10",
                        help="Comma-separated K values for P@K and NDCG@K (default: 5,10)")
    parser.add_argument("--variant", type=str, default=None,
                        help="Single variant to run: full, text, deps, topics, tfidf, topic_tag. "
                             "Default: run all.")
    parser.add_argument("--out", type=str, default=None,
                        help="Save results JSON to this path")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-query output, print summary only")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    repos_path = Path(args.repos) if args.repos else None
    k_values = [int(x.strip()) for x in args.k.split(",")]
    all_variants = ["full", "text", "deps", "topics", "tfidf", "topic_tag"]
    variants = [args.variant] if args.variant else all_variants

    results = evaluate(
        data_dir=data_dir,
        repos_path=repos_path,
        k_values=k_values,
        variants=variants,
        verbose=not args.quiet,
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {out_path}")

if __name__ == "__main__":
    main()