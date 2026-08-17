## CS499 Final 

This subproject is a sandbox to experiment with smaller, high-quality curated GitHub repo datasets for the similarity engine from the main project.

### Goals

- Focus on **small, hand-picked corpora** per ecosystem (e.g. frontend frameworks, C/C++, Python, Java, SQL).
- Tune **feature extraction** (README, topics, manifests, ecosystems) and **similarity weights** for more meaningful results.
- Validate quality on known anchors (e.g. `facebook/react`) before scaling back up.

### Layout

- `datasets/` – curated repo lists per experiment (not included in the code-only submission zip).
- `datasets_expanded/` – generated artifacts (repos JSON, embeddings, FAISS index) (not included in the code-only submission zip).
- `scripts/` – thin wrappers around the main `src` pipeline configured to use curated or expanded datasets.

### How to use

1. Add a curated repo list under `datasets/` (a plain text file with one `owner/repo` per line).
2. Run an experiment script in `scripts/` to:
   - collect data
   - build embeddings
   - build a FAISS index
3. Use the main app or CLI pointed at the small index to inspect results.

#### Scale-up script

From the project root:

```bash
bash pt2/scripts/run_expanded_pipeline.sh [optional_path_to_repo_list]
```

By default it reads `data/repo_list_small.txt` and writes outputs to `pt2/datasets_expanded/`.

