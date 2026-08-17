#Used my friend's public github repository to test it on since it is a niche/non popular repo

"""
Add one GitHub repo to the index: fetch its data, append to repos.json,
then regenerate embeddings and FAISS index.

Usage:
  python -m src.add_repo_to_index Muralikinti/ChessBot
  python -m src.add_repo_to_index https://github.com/Muralikinti/ChessBot
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPOS_PATH = DATA_DIR / "repos.json"

def parse_repo(s: str) -> tuple[str, str]:
    """Return (owner, name) from 'owner/name' or GitHub URL."""
    s = s.strip().rstrip("/")
    m = re.match(r"(?:https?://)?(?:www\.)?github\.com/([^/]+)/([^/]+?)(?:\.git)?$", s)
    if m:
        return m.group(1), m.group(2)
    if "/" in s:
        parts = s.split("/", 1)
        return parts[0].strip(), parts[1].strip()
    raise ValueError(f"Use owner/name or GitHub URL: {s}")

def main():
    parser = argparse.ArgumentParser(description="Add one repo to the index and rebuild.")
    parser.add_argument("repo", help="owner/name or URL, e.g. Muralikinti/ChessBot or https://github.com/Muralikinti/ChessBot")
    parser.add_argument("--no-rebuild", action="store_true", help="Only add to repos.json; do not run embeddings or FAISS build")
    args = parser.parse_args()

    try:
        owner, name = parse_repo(args.repo)
    except ValueError as e:
        print(e)
        sys.exit(1)

    full_name = f"{owner}/{name}"
    if not REPOS_PATH.exists():
        print(f"Not found: {REPOS_PATH}. Run data collection first.")
        sys.exit(1)

    from src.collect_repo_data import collect_one

    print(f"Fetching {full_name}...")
    try:
        row = collect_one(owner, name, include_stars=False)
    except Exception as e:
        print(f"Failed to fetch: {e}")
        sys.exit(1)

    with open(REPOS_PATH, encoding="utf-8") as f:
        repos = json.load(f)
    existing = {r["repo"] for r in repos}
    if full_name in existing:
        print(f"{full_name} is already in the index.")
        sys.exit(0)

    repos.append(row)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPOS_PATH, "w", encoding="utf-8") as f:
        json.dump(repos, f, indent=2, ensure_ascii=False)
    print(f"Added {full_name} to repos.json ({len(repos)} repos total).")

    if args.no_rebuild:
        print("Skipping rebuild. Run: python -m src.embeddings && python -m src.similarity build")
        return

    print("Regenerating embeddings...")
    r = subprocess.run(
        [sys.executable, "-m", "src.embeddings", "--repos", str(REPOS_PATH), "--out-dir", str(DATA_DIR)],
        cwd=str(PROJECT_ROOT),
    )
    if r.returncode != 0:
        print("Embeddings failed.")
        sys.exit(r.returncode)
    print("Rebuilding FAISS index...")
    r = subprocess.run(
        [sys.executable, "-m", "src.similarity", "build", "--data-dir", str(DATA_DIR)],
        cwd=str(PROJECT_ROOT),
    )
    if r.returncode != 0:
        print("FAISS build failed.")
        sys.exit(r.returncode)
    print("Done. You can query Muralikinti/ChessBot in the app.")

if __name__ == "__main__":
    main()