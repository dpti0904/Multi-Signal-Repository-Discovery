# Multi-Signal Repository Discovery

Find GitHub repositories similar to a given repo using README semantics, dependency overlap, and star-based collaborative filtering.


## Setup

```bash
python -m venv venv

pip install -r requirements.txt
```

## Project layout 

- `src/` – data collection, embedding, similarity, and FAISS index code
- `app.py` 

## How to run

This submission is a **code-only compressed project** (no `venv/`, large datasets, or prebuilt FAISS artifacts). If you want to run the Streamlit app, you must first build the index locally.

### Build the index (recommended: ~1000 repos)

From the project root:

```bash
# Collect data + build embeddings + build FAISS index into pt2/datasets_expanded/
bash pt2/scripts/run_expanded_pipeline.sh
```

This produces:

- `pt2/datasets_expanded/repos_expanded.json`
- `pt2/datasets_expanded/faiss.index` (and embedding artifacts)

### Launch the app

7. **Launch the Streamlit UI**:
   ```bash
   streamlit run app.py
   ```
   Enter a repo (e.g. `facebook/react`) and see ranked similar repos with score breakdown.
