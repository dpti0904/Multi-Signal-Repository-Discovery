"""
Week 3: Combined similarity (README + dependencies) and FAISS retrieval.
Loads embeddings and repos, builds FAISS index, returns ranked similar repos.
Supports querying any GitHub repo (not just indexed): fetches README + deps on the fly, embeds, searches.

Usage:
  python -m src.similarity build
  python -m src.similarity query facebook/react --k 10
  python -m src.similarity query https://github.com/someone/AnyRepo --k 10
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import faiss

_SBERT_MODEL = None
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

def _get_model():
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _SBERT_MODEL = SentenceTransformer(SBERT_MODEL_NAME)
    return _SBERT_MODEL

def parse_repo(s: str) -> str:
    """Normalize to owner/name. Accepts URL or owner/name."""
    s = s.strip().rstrip("/")
    m = re.match(r"(?:https?://)?(?:www\.)?github\.com/([^/]+)/([^/]+?)(?:\.git)?$", s)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    if "/" in s:
        return s
    return s

W_TEXT = 0.45
W_DEPS = 0.25
W_TOPICS = 0.15
W_ECOSYSTEM = 0.07
W_LANGUAGE = 0.03
W_PROJECT_TYPE = 0.05

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
    "repository", "github", "code", "example", "examples", "based", "simple",
    "support", "file", "files", "add", "run", "make", "build", "install",
    "see", "other", "some", "such", "only", "very", "well", "need", "way",

    "framework", "library", "tool", "tools", "open", "source", "fast",
    "easy", "free", "available", "official", "release", "version", "package",
    "module", "implementation", "provides", "allows", "enables", "built",
    "written", "language", "application", "app", "system", "platform",
    "data", "api", "web", "server", "client", "list", "star", "read",
    "write", "create", "update", "delete", "work", "works", "working",
    "feature", "features", "documentation", "doc", "docs", "guide", "help",
    "note", "notes", "include", "includes", "including", "plus", "via",
    "without", "within", "across", "service", "services", "interface",
    "best", "high", "low", "full", "small", "large", "many", "most",
    "following", "given", "current", "multiple", "single", "third", "party",
}

def _extract_keywords(text: str, n: int = 15) -> set[str]:
    """
    Extract top-N keywords from text as synthetic topics when GitHub topics are empty.
    Simple frequency-based extraction — no extra dependencies needed.
    Filters stopwords and short tokens; favors multi-char technical terms.
    """
    words = re.findall(r'\b[a-z][a-z0-9_-]{2,}\b', text.lower())
    filtered = [w for w in words if w not in _STOPWORDS]
    counts = Counter(filtered)
    return {w for w, _ in counts.most_common(n)}

def _get_topics(repo_dict: dict, readme_chars: int = 2000) -> set[str]:
    """
    Return GitHub topics for a repo. If empty, fall back to keyword extraction
    from description + README so topic Jaccard never collapses to zero just
    because a repo has no GitHub topics set.
    """
    topics = set(repo_dict.get("topics") or [])
    if topics:
        return topics
    desc = repo_dict.get("description") or ""
    readme = (repo_dict.get("readme_text") or "")[:readme_chars]
    return _extract_keywords(desc + " " + readme, n=15)

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def _norm_deps(deps) -> set:
    """Normalize dependency names (lowercase) so NumPy vs numpy matches."""
    if not deps:
        return set()
    return {str(x).strip().lower() for x in deps if x}

def _infer_ecosystems(ecosystems, language: str) -> set[str]:
    """
    Fallback when manifest parsing yields no ecosystems.
    Keeps the ecosystem signal alive (weak tiebreaker) even for repos that
    don't declare dependencies in standard files.
    """
    eco = {str(x).strip().lower() for x in (ecosystems or []) if x}
    if eco:
        return eco
    lang = (language or "").strip().lower()
    if not lang:
        return set()
    if lang in {"python"}:
        return {"python"}
    if lang in {"javascript", "typescript"}:
        return {"node"}
    if lang in {"go"}:
        return {"go"}
    if lang in {"rust"}:
        return {"rust"}
    if lang in {"java"}:
        return {"java"}
    if lang in {"c", "c++", "cpp"}:
        return {"c-cpp"}
    if lang in {"c#", "csharp"}:
        return {"dotnet"}
    if lang in {"ruby"}:
        return {"ruby"}
    if lang in {"php"}:
        return {"php"}
    return set()

def load_data(data_dir: Path, repos_path: Optional[Path] = None):
    data_dir = Path(data_dir)
    repos_file = Path(repos_path) if repos_path else (data_dir / "repos.json")
    with open(repos_file, encoding="utf-8") as f:
        repos = json.load(f)
    embeddings = np.load(data_dir / "embeddings.npy")
    with open(data_dir / "repo_index.json", encoding="utf-8") as f:
        repo_index = json.load(f)
    return repos, embeddings, repo_index

def build_index(data_dir: Path, save: bool = True, repos_path: Optional[Path] = None):
    repos, embeddings, repo_index = load_data(data_dir, repos_path)
    faiss.normalize_L2(embeddings)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings.astype(np.float32))
    if save:
        data_dir = Path(data_dir)
        faiss.write_index(index, str(data_dir / "faiss.index"))
        print(f"Saved FAISS index to {data_dir / 'faiss.index'}")
    return index, repos, repo_index, embeddings

def load_index(data_dir: Path, repos_path: Optional[Path] = None):
    data_dir = Path(data_dir)
    index = faiss.read_index(str(data_dir / "faiss.index"))
    repos, _, repo_index = load_data(data_dir, repos_path)
    return index, repos, repo_index

def _candidate_k(k: int, n: int) -> int:
    """
    Retrieve more than k from FAISS, then rerank by hybrid score.
    For small corpora this is fast and improves final quality.
    """
    if n <= 0:
        return k
    base = max(200, k * 50)
    return max(k, min(base, 1000, n))

def _score_pair(q: dict, cand: dict, text_score: float) -> dict:
    """
    Compute hybrid similarity scores between a query repo and a candidate repo.
    Extracted as a shared helper so get_similar and get_similar_any_repo
    use identical scoring logic.

    q and cand must each have: dependencies, topics, language, ecosystems,
    project_type, readme_text, description, repo (name).
    """
    q_deps = _norm_deps(q.get("dependencies") or [])
    cand_deps = _norm_deps(cand.get("dependencies") or [])

    q_topics = _get_topics(q)
    cand_topics = _get_topics(cand)

    q_lang = (q.get("language") or "").lower()
    cand_lang = (cand.get("language") or "").lower()

    q_eco = _infer_ecosystems(q.get("ecosystems") or [], q_lang)
    cand_eco = _infer_ecosystems(cand.get("ecosystems") or [], cand_lang)

    q_type = (q.get("project_type") or "").lower()
    cand_type = (cand.get("project_type") or "").lower()

    j_dep = jaccard(q_deps, cand_deps)
    j_topic = jaccard(q_topics, cand_topics)
    j_eco = jaccard(q_eco, cand_eco)
    lang_match = 1.0 if (q_lang and cand_lang and q_lang == cand_lang) else 0.0
    type_match = 1.0 if (q_type and cand_type and q_type == cand_type) else 0.0

    combined = (
        W_TEXT * float(text_score)
        + W_DEPS * j_dep
        + W_TOPICS * j_topic
        + W_ECOSYSTEM * j_eco
        + W_LANGUAGE * lang_match
        + W_PROJECT_TYPE * type_match
    )

    if q_lang and cand_lang and q_lang != cand_lang:
        if float(text_score) > 0.5:
            combined *= 0.85
        else:
            combined *= 0.7

    desc = (cand.get("description") or "").strip()
    if not desc:
        readme = (cand.get("readme_text") or "").strip()
        desc = readme.splitlines()[0].strip() if readme else ""

    return {
        "repo": cand.get("repo", ""),
        "summary": desc,
        "text_score": float(text_score),
        "dep_score": j_dep,
        "topic_score": j_topic,
        "ecosystem_score": j_eco,
        "language_match": lang_match,
        "combined_score": combined,
        "project_type": cand_type,
    }

def _apply_diversity_penalty(results: list, query_repo: str, q_type: str) -> list:
    """Downweight awesome-list repos when the query is a specific project."""
    q_is_awesome = q_type == "awesome-list" or "awesome" in query_repo.lower()
    for r in results:
        pt = r.get("project_type") or ""
        name_lower = r.get("repo", "").lower()
        if (pt == "awesome-list" or "awesome" in name_lower) and not q_is_awesome:
            r["combined_score"] *= 0.6
    return results

def get_similar(
    query_repo: str,
    k: int,
    data_dir: Path,
    index=None,
    repos=None,
    repo_index=None,
    embeddings=None,
    repos_path: Optional[Path] = None,
):
    data_dir = Path(data_dir)
    if repos is None or repo_index is None or embeddings is None:
        repos, embeddings, repo_index = load_data(data_dir, repos_path)
    if index is None:
        index = faiss.read_index(str(data_dir / "faiss.index"))

    try:
        q_idx = repo_index.index(query_repo)
    except ValueError:
        return None, f"Repo not in index: {query_repo}"

    q_embed = embeddings[q_idx : q_idx + 1].astype(np.float32)
    faiss.normalize_L2(q_embed)

    k_cand = _candidate_k(k, len(repo_index))
    k_actual = min(k_cand + 1, len(repo_index))
    scores, indices = index.search(q_embed, k_actual)

    q_dict = repos[q_idx]
    results = []
    for text_score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        if repo_index[idx] == query_repo:
            continue
        scored = _score_pair(q_dict, repos[idx], text_score)
        results.append(scored)

    q_type = (q_dict.get("project_type") or "").lower()
    results = _apply_diversity_penalty(results, query_repo, q_type)
    results.sort(key=lambda r: r["combined_score"], reverse=True)
    return results[:k], None

def get_similar_any_repo(
    query_repo: str,
    k: int,
    data_dir: Path,
    index=None,
    repos=None,
    repo_index=None,
    embeddings=None,
    repos_path: Optional[Path] = None,
):
    """
    Find repos similar to query_repo. If query_repo is in the index, use it.
    If not, fetch its README + deps from GitHub, embed, and search the index.
    query_repo can be owner/name or a GitHub URL.
    """
    data_dir = Path(data_dir)
    query_repo = parse_repo(query_repo)
    if repos is None or repo_index is None or embeddings is None:
        repos, embeddings, repo_index = load_data(data_dir, repos_path)
    if index is None:
        index = faiss.read_index(str(data_dir / "faiss.index"))

    if query_repo in repo_index:
        return get_similar(
            query_repo, k, data_dir,
            index=index, repos=repos, repo_index=repo_index,
            embeddings=embeddings, repos_path=repos_path,
        )

    try:
        from src.collect_repo_data import collect_one
    except ImportError:
        from collect_repo_data import collect_one

    if "/" not in query_repo:
        return None, f"Use owner/name or GitHub URL: {query_repo}"

    owner, name = query_repo.split("/", 1)
    try:
        row = collect_one(owner, name, include_stars=False)
    except Exception as e:
        return None, f"Could not fetch repo: {e}"

    try:
        from src.embedding_text import build_embedding_text
    except ImportError:
        from embedding_text import build_embedding_text

    text = build_embedding_text(row)
    model = _get_model()
    q_embed = model.encode([text], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(q_embed)

    k_cand = _candidate_k(k, len(repo_index))
    k_actual = min(k_cand, len(repo_index))
    scores, indices = index.search(q_embed, k_actual)

    row["repo"] = query_repo

    results = []
    for text_score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        scored = _score_pair(row, repos[idx], text_score)
        results.append(scored)

    q_type = (row.get("project_type") or "").lower()
    results = _apply_diversity_penalty(results, query_repo, q_type)
    results.sort(key=lambda r: r["combined_score"], reverse=True)
    return results[:k], None

def main():
    parser = argparse.ArgumentParser(description="Build FAISS index or query similar repos.")
    parser.add_argument("command", choices=["build", "query"], help="build index or query")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory with embeddings.npy, repo_index.json")
    parser.add_argument("--repos", type=str, default=None, help="Path to repos JSON (default: <data-dir>/repos.json).")
    parser.add_argument("--k", type=int, default=10, help="Number of similar repos to return (query only)")
    parser.add_argument("repo", nargs="?", help="Query repo owner/name for 'query'")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    repos_path = Path(args.repos) if args.repos else None
    repos_file = repos_path or (data_dir / "repos.json")
    if not repos_file.exists():
        print(f"Missing repos file: {repos_file}. Run data collection first.")
        return
    if not (data_dir / "embeddings.npy").exists():
        print("Missing embeddings.npy. Run: python -m src.embeddings --out-dir <data-dir>")
        return

    if args.command == "build":
        build_index(data_dir, repos_path=repos_path)
        return

    if args.command == "query":
        if not args.repo:
            print("Usage: python -m src.similarity query owner/repo [--k 10]")
            return
        if not (data_dir / "faiss.index").exists():
            print("Building FAISS index first...")
            build_index(data_dir, repos_path=repos_path)
        results, err = get_similar_any_repo(args.repo, args.k, data_dir, repos_path=repos_path)
        if err:
            print(err)
            return
        print(f"Top {args.k} similar to {args.repo}:")
        for i, r in enumerate(results, 1):
            t = r.get("topic_score", 0)
            e = r.get("ecosystem_score", 0)
            l = r.get("language_match", 0)
            print(
                f"  {i}. {r['repo']}  "
                f"(text={r['text_score']:.3f} dep={r['dep_score']:.3f} "
                f"topic={t:.3f} eco={e:.3f} lang={l:.0f} "
                f"combined={r['combined_score']:.3f})"
            )

if __name__ == "__main__":
    main()