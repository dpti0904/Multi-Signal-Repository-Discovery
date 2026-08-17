"""
Discover GitHub repository identifiers via the Search API.
Uses a broad range of queries (languages, topics, scales) for diverse corpus.
Writes owner/name (one per line) to a file for use with collect_repo_data.py.

Usage:
  python -m src.discover_repos --target 3000 --out data/repo_list.txt
  python -m src.discover_repos --target 5000

Requires: GITHUB_TOKEN in environment (search is rate-limited to 30 requests/min).
"""

import argparse
import os
import time
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GITHUB_API = "https://api.github.com"

def get_headers():
    token = os.environ.get("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    return headers

def search_repos(query: str, per_page: int = 100, page: int = 1) -> list[dict]:
    """Return list of repo items from search (each has 'full_name', 'stargazers_count', etc.)."""
    url = f"{GITHUB_API}/search/repositories"
    r = requests.get(
        url,
        headers=get_headers(),
        params={"q": query, "sort": "stars", "order": "desc", "per_page": per_page, "page": page},
    )
    if r.status_code != 200:
        raise RuntimeError(f"Search failed: {r.status_code} - {r.text[:200]}")
    data = r.json()
    return data.get("items", [])

def discover(target: int = 3000) -> list[str]:
    """
    Use a broad range of search queries for diverse corpus (stakeholder-driven).
    Covers: languages, web/fullstack, ML, games, student projects, APIs, DevOps, etc.
    """
    queries = [

        "stars:>500 language:Python",
        "stars:>500 language:JavaScript",
        "stars:>300 language:Go",
        "topic:machine-learning stars:>200",

        "stars:100..500",
        "stars:50..200",
        "topic:web-framework stars:>50",
        "topic:fullstack stars:>30",
        "topic:api stars:>100",
        "topic:nodejs stars:>50",
        "topic:react stars:>50",
        "topic:django stars:>30",
        "topic:flask stars:>30",

        "language:Python stars:>50",
        "language:JavaScript stars:>50",
        "language:TypeScript stars:>50",
        "language:Go stars:>30",
        "language:Rust stars:>30",
        "language:Java stars:>50",
        "language:C stars:>30",
        "language:C++ stars:>30",
        "language:C# stars:>30",
        "language:Ruby stars:>20",
        "language:PHP stars:>30",
        "language:Swift stars:>20",
        "language:Kotlin stars:>30",
        "language:R stars:>20",
        "language:Scala stars:>20",
        "language:Lua stars:>10",
        "language:Shell stars:>20",
        "language:HTML stars:>10",

        "stars:20..100",
        "stars:10..50",
        "stars:5..30",
        "topic:student-project",
        "topic:web-game",
        "topic:tutorial stars:>10",
        "topic:fullstack stars:>5",
        "topic:internship",
        "topic:learning",
        "topic:beginner-project",
        "topic:cli-tool stars:>10",
        "topic:devops stars:>20",
    ]
    seen = set()
    full_names = []

    for q in queries:
        if len(full_names) >= target:
            break
        for page in range(1, 11):
            if len(full_names) >= target:
                break
            try:
                items = search_repos(q, per_page=100, page=page)
            except Exception as e:
                print(f"Query '{q}' page {page} failed: {e}")
                break
            if not items:
                break
            for repo in items:
                name = repo.get("full_name")
                if name and name not in seen:
                    seen.add(name)
                    full_names.append(name)
            print(f"  {q} page {page}: {len(full_names)} total so far")
            time.sleep(2.1)

    return full_names[:target]

def main():
    parser = argparse.ArgumentParser(description="Discover repo list via GitHub Search for data collection.")
    parser.add_argument("--target", type=int, default=3000, help="Target number of repos (default 3000)")
    parser.add_argument("--out", type=str, default="data/repo_list.txt", help="Output file (one owner/name per line)")
    args = parser.parse_args()

    if not os.environ.get("GITHUB_TOKEN"):
        print("Warning: GITHUB_TOKEN not set. Search is rate-limited to 30 req/min.")

    print(f"Discovering ~{args.target} repositories...")
    names = discover(target=args.target)
    print(f"Got {len(names)} unique repos.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()