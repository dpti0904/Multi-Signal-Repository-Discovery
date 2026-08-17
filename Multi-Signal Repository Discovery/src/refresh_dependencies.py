"""
Refresh dependency + ecosystem fields for an already-collected repos JSON.

Why: if repos were collected before improving manifest parsing (e.g. pyproject.toml PEP 621),
many entries may have empty dependencies/ecosystems. This script re-fetches only the manifest
files (via GitHub Contents API) and rewrites the dependencies/ecosystems fields in-place.

It does NOT re-fetch README or topics/metadata (so it's cheaper than a full recollection).

Usage:
  # Refresh expanded corpus deps (resumable)
  python -m src.refresh_dependencies --repos pt2/datasets_expanded/repos_expanded.json --save-every 25

  # Limit to first 50 repos for a quick sanity check
  python -m src.refresh_dependencies --repos pt2/datasets_expanded/repos_expanded.json --limit 50
"""

import argparse
import json
import time
from pathlib import Path

from src.collect_repo_data import get_dependencies

def main():
    parser = argparse.ArgumentParser(description="Refresh dependencies/ecosystems for existing repos JSON.")
    parser.add_argument("--repos", required=True, help="Path to repos JSON (array of repo objects).")
    parser.add_argument("--out", default=None, help="Output path (default: overwrite --repos).")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N repos (0 = all).")
    parser.add_argument("--resume", action="store_true", help="Skip repos that already have non-empty dependencies.")
    parser.add_argument("--save-every", type=int, default=0, help="Save progress every N processed repos.")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep seconds between repos to be gentle on API.")
    args = parser.parse_args()

    repos_path = Path(args.repos)
    if not repos_path.exists():
        raise FileNotFoundError(f"Not found: {repos_path}")
    out_path = Path(args.out) if args.out else repos_path

    with open(repos_path, encoding="utf-8") as f:
        repos = json.load(f)
    if not isinstance(repos, list):
        raise ValueError("repos JSON must be a list of repo objects")

    limit = args.limit or len(repos)
    processed = 0
    updated = 0
    skipped = 0

    for i, r in enumerate(repos[:limit]):
        full = r.get("repo") or ""
        if "/" not in full:
            skipped += 1
            continue
        if args.resume and (r.get("dependencies") or []):
            skipped += 1
            continue

        owner, name = full.split("/", 1)
        try:
            deps, ecosystems = get_dependencies(owner, name)
        except Exception:
            skipped += 1
            continue

        r["dependencies"] = sorted(deps)
        r["ecosystems"] = sorted(ecosystems)
        processed += 1
        if deps or ecosystems:
            updated += 1

        if args.save_every and processed and processed % args.save_every == 0:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(repos[:limit], f, indent=2, ensure_ascii=False)
            print(f"Saved progress: processed={processed} updated={updated} skipped={skipped}")

        time.sleep(args.sleep)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(repos[:limit], f, indent=2, ensure_ascii=False)
    print(f"Done. processed={processed} updated={updated} skipped={skipped} -> {out_path}")

if __name__ == "__main__":
    main()
