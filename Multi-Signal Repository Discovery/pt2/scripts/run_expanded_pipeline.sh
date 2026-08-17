#!/usr/bin/env bash
#  Run the improved pipeline on the larger repo_list_small corpus.
# Run from project root: bash pt2/scripts/run_expanded_pipeline.sh
# Keeps curated pt2/datasets/ untouched; writes to pt2/datasets_expanded/.

set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

EXPANDED_DIR="pt2/datasets_expanded"
REPOS_LIST="${1:-data/repo_list_small.txt}"

echo "=== 1. Create output dir ==="
mkdir -p "$EXPANDED_DIR"

echo "=== 2. Collect repo data (this may take a while; use --resume to continue if interrupted) ==="
python -m src.collect_repo_data \
  --list "$REPOS_LIST" \
  --out "$EXPANDED_DIR/repos_expanded.json" \
  --resume \
  --save-every 50

echo "=== 3. Build embeddings ==="
python -m src.embeddings \
  --repos "$EXPANDED_DIR/repos_expanded.json" \
  --out-dir "$EXPANDED_DIR"

echo "=== 4. Build FAISS index ==="
python -m src.similarity build \
  --data-dir "$EXPANDED_DIR" \
  --repos "$EXPANDED_DIR/repos_expanded.json"

echo "=== Done. Query with: ==="
echo "  python -m src.similarity query <owner/repo> --k 10 --data-dir $EXPANDED_DIR --repos $EXPANDED_DIR/repos_expanded.json"
