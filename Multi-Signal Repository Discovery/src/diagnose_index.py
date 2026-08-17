"""
Diagnostic: print dependencies, topics, and README length for repos in a JSON file.
Use this to verify your crawler is storing the right fields (especially while
the 1000-repo pipeline runs). If topics/dependencies/readme_text are missing
or empty, fix the crawler before continuing.

Usage:
  python -m src.diagnose_index pt2/datasets/repos_core.json
  python -m src.diagnose_index data/repos.json --limit 5

  # Spot-check what the pipeline is storing (first 3 repos, quick format):
  python -m src.diagnose_index pt2/datasets_expanded/repos_expanded.json --spot-check
"""

import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Print deps/topics/readme for repos to debug 0.000 scores or spot-check pipeline output."
    )
    parser.add_argument("repos_json", type=str, help="Path to repos.json (e.g. pt2/datasets_expanded/repos_expanded.json)")
    parser.add_argument("--limit", type=int, default=0, help="Max repos to print (0 = all). Ignored if --spot-check.")
    parser.add_argument(
        "--spot-check",
        action="store_true",
        help="Quick check: first 3 repos only, with topics/dependencies/readme_length. Use this on pipeline output while it runs.",
    )
    args = parser.parse_args()

    path = Path(args.repos_json)
    if not path.exists():
        print(f"Not found: {path}")
        return
    with open(path, encoding="utf-8") as f:
        repos = json.load(f)

    if not isinstance(repos, list):
        print("repos_json is not a list of repo objects")
        return

    if args.spot_check:
        n = min(3, len(repos))
        print(f"Spot check: first {n} repos in {path} (total so far: {len(repos)})\n")
        for r in repos[:n]:
            full_name = r.get("repo") or r.get("full_name")
            print(full_name if full_name else "MISSING repo/full_name")
            topics = r.get("topics")
            print("  topics:", topics if topics is not None else "MISSING")
            deps = r.get("dependencies")
            print("  dependencies:", deps if deps is not None else "MISSING")
            readme = r.get("readme_text")
            print("  readme_length:", len(readme) if readme else ("MISSING" if readme is None else 0))
            print()
        if len(repos) >= 3:
            print("If topics/dependencies are MISSING or empty, or readme_length is 0, stop the pipeline, fix the crawler, and restart.")
            print("If they look populated, let the pipeline finish — the bug is in scoring, not collection.")
        return

    limit = args.limit or len(repos)
    n_deps_empty = 0
    n_topics_empty = 0
    n_readme_empty = 0
    for i, r in enumerate(repos[:limit]):
        full_name = r.get("repo") or r.get("full_name") or "?"
        deps = r.get("dependencies")
        topics = r.get("topics")
        readme = r.get("readme_text") or ""
        if not isinstance(deps, list):
            deps = []
        if not isinstance(topics, list):
            topics = []
        if not deps:
            n_deps_empty += 1
        if not topics:
            n_topics_empty += 1
        if not readme or not readme.strip():
            n_readme_empty += 1
        print(f"{full_name}")
        print(f"  deps ({len(deps)}): {deps[:8]}{'...' if len(deps) > 8 else ''}")
        print(f"  topics ({len(topics)}): {topics[:10]}{'...' if len(topics) > 10 else ''}")
        print(f"  readme_length: {len(readme)}")
        print()

    total = min(limit, len(repos))
    print(f"--- Summary (first {total} repos) ---")
    print(f"Repos with empty dependencies: {n_deps_empty}/{total}")
    print(f"Repos with empty topics: {n_topics_empty}/{total}")
    print(f"Repos with empty/missing README: {n_readme_empty}/{total}")
    if n_deps_empty == total:
        print("  -> All deps empty: crawler may not be finding manifest files, or repos have none.")
    if n_topics_empty == total:
        print("  -> All topics empty: ensure GitHub token has repo scope and API returns topics.")
    if n_readme_empty == total:
        print("  -> All READMEs empty: crawler may not be fetching or storing readme_text.")

if __name__ == "__main__":
    main()