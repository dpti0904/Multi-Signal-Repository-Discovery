# Multi-Signal Repository Discovery

A personal project exploring how to discover GitHub repositories that are functionally similar to a given project using more than just text similarity.

The idea is simple: instead of searching by keyword alone, I wanted a system that combines multiple signals to answer a question like, “What other repositories are conceptually close to this project?” I built a pipeline that collects GitHub metadata, embeds README and description text, evaluates dependency overlap, topic similarity, and repository ecosystem signals, then ranks likely matches with a FAISS vector index.
---

## Why I built this

I wanted to experiment with a system that can answer a very real software discovery problem:

- If I’m looking at a repository like `facebook/react`, what are the closest comparable repositories?
- If a project is niche or not widely known, can I still find similar work using metadata and technical structure?
- Can a hybrid approach outperform a single text-only similarity baseline?

The answer turned out to be a multi-signal ranking system that blends:

- semantic similarity from README and description text
- dependency overlap from manifests like `package.json`, `requirements.txt`, `pyproject.toml`, `go.mod`, etc.
- topic overlap from GitHub tags and keyword fallback
- ecosystem and language signals
- a vector database for fast retrieval

The result is a small but useful repo-similarity engine that feels closer to real developer discovery than a simple keyword matcher.

---

## What the project does

At a high level, the project does the following:

1. Finds a set of GitHub repositories to index
2. Pulls metadata for each repo from the GitHub API
3. Extracts text from READMEs and project descriptions
4. Extracts dependency fingerprints from manifests
5. Builds semantic embeddings with Sentence-BERT
6. Stores the embeddings in a FAISS index
7. Ranks similar repos using a hybrid score
8. Exposes the result in a Streamlit UI
9. Optionally explains why repos are similar using Groq

This means the project is not just a demo; it is a full pipeline from discovery to retrieval.

---

## Project architecture

### 1. Repo discovery

The repository search layer is built in `src/discover_repos.py`.

It uses GitHub’s repository search API with a broad range of queries to collect a diverse corpus of candidate repositories. It intentionally searches by:

- language
- stars
- topics like `machine-learning`, `web-framework`, `api`, `devops`, etc.
- ecosystem mix across Python, JavaScript, Go, Rust, Java, and more

This gives a wider and more diverse index than just searching one language or framework.

### 2. Data collection

The core data collection is handled in `src/collect_repo_data.py`.

For each repository, it fetches:

- README content
- project description
- GitHub topics
- main language
- stars/forks/watchers
- license, project size, metadata
- dependency manifests
- ecosystem tags
- optional stargazer data

It also parses common dependency files like:

- `package.json`
- `requirements.txt`
- `pyproject.toml`
- `setup.py`
- `pom.xml`
- `build.gradle`
- `go.mod`
- `Cargo.toml`
- `Gemfile`
- `CMakeLists.txt`
- `Makefile`

This is important because repo similarity is often stronger when technical overlap is considered, not just README similarity.

### 3. Embedding layer

The text-to-embedding flow lives in `src/embedding_text.py` and `src/embeddings.py`.

The repository text used for embedding is composed of:

- description
- GitHub topics
- the beginning of the README

This makes the embedding more informative than README text alone while keeping it lightweight and consistent across the corpus.

The embeddings are generated with `sentence-transformers` using `all-MiniLM-L6-v2` and then saved as `embeddings.npy` with a matching `repo_index.json`.

### 4. Similarity and retrieval

The actual hybrid ranking logic is in `src/similarity.py`.

It uses a weighted combination of signals:

- text similarity: 45%
- dependency overlap: 25%
- topic overlap: 15%
- ecosystem similarity: 7%
- language match: 3%
- project type match: 5%

The score is not just cosine similarity over embeddings. It is a hybrid score that re-ranks vector search results using multiple repository-level features. That is the key part of this project: semantic similarity alone is useful, but technical context adds a lot of signal.

### 5. FAISS index

After embeddings are built, the system creates a FAISS index with normalized vectors and performs fast nearest-neighbor search.

This allows the app to retrieve candidate repositories quickly, then rerank them based on the custom hybrid score.

### 6. Streamlit app and AI explanation layer

The app is in `app.py`.

It lets a user enter a repository in one of these formats:

- `owner/repo`
- GitHub URL

Then it returns similar repositories with details on:

- text similarity
- dependency score
- topic score
- ecosystem score
- language match
- total combined score

There is also optional Groq-based explanation support in `src/groq_explain.py`. This is useful for explaining why two repos are considered similar in human language, especially when the overlap is not obvious.

### 7. Evaluation framework

The evaluation work is in `src/evaluate.py`.

It measures similarity quality using common retrieval metrics such as:

- Precision@K
- MRR
- NDCG@K

This project is designed to compare multiple variants, including:

- full hybrid approach
- text-only baseline
- dependency-only baseline
- topic-only baseline
- TF-IDF baseline

That helps validate that the combined signal is actually better than simpler alternatives.

---

## Repository structure

- `app.py` — Streamlit frontend for searching and exploring repo matches
- `src/` — all core logic for discovery, collection, embedding, scoring, and explanation
- `pt2/` — experimental pipeline and curated dataset scaffolding
- `requirements.txt` — Python dependencies
- `requirements-colab.txt` — lightweight dependency set for notebook/Colab workflows
- `.env` — local environment file for tokens such as GitHub and Groq keys

Key files:

- `src/discover_repos.py` — discovers GitHub repos to populate a corpus
- `src/collect_repo_data.py` — fetches metadata and dependency fingerprints
- `src/embedding_text.py` — defines the text used for embeddings
- `src/embeddings.py` — generates Sentence-BERT embeddings
- `src/similarity.py` — hybrid scoring and FAISS retrieval
- `src/groq_explain.py` — LLM explanation of why repos match
- `src/evaluate.py` — evaluation and baseline comparison
- `src/add_repo_to_index.py` — adds a repo to the corpus and rebuilds the index

---

## How it works in practice

The full flow typically looks like this:

1. Create a repo list with `src/discover_repos.py`
2. Collect repo metadata and dependency data with `src/collect_repo_data.py`
3. Build embeddings with `src/embeddings.py`
4. Build a FAISS index with `src/similarity.py`
5. Query the app or CLI with some repository name
6. View the most similar repos with a breakdown of the signal contributions

A query is not just “find similar text.” It is “find repos with overlapping technical structure and purpose.”

---

## Setup

This project is built for Python 3.10+.

### 1. Create a virtual environment

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root if you want to use the GitHub API and Groq explanations.

```env
GITHUB_TOKEN=your_github_token
GROQ_API_KEY=your_groq_api_key
```

Notes:

- `GITHUB_TOKEN` is strongly recommended for GitHub API rate limits
- `GROQ_API_KEY` is optional and only needed for AI-generated explanations in the UI

---

## Data pipeline

If you want to build the project from scratch, here is the recommended flow.

### Option A: Discover a new repo list

```bash
python -m src.discover_repos --target 3000 --out data/repo_list.txt
```

This generates a plain-text list of `owner/repo` entries.

### Option B: Collect repository metadata

```bash
python -m src.collect_repo_data --list data/repo_list.txt --out data/repos.json --resume --save-every 50
```

This fetches metadata and checks common dependency manifests across the repo list.

### Option C: Build embeddings

```bash
python -m src.embeddings --repos data/repos.json --out-dir data
```

This creates:

- `data/embeddings.npy`
- `data/repo_index.json`

### Option D: Build the FAISS index

```bash
python -m src.similarity build --data-dir data --repos data/repos.json
```

This creates the searchable vector index used by the app.

---

## Running the app

From the project root:

```bash
streamlit run app.py
```

Then enter a repository such as:

- `facebook/react`
- `pallets/flask`
- `karpathy/micrograd`
- `microsoft/vscode`

The app shows ranked similar repositories and a score breakdown for each result.

---

## Using the CLI directly

You can also query the similarity engine directly from the terminal.

```bash
python -m src.similarity query facebook/react --k 10 --data-dir data --repos data/repos.json
```

Or with a GitHub URL:

```bash
python -m src.similarity query https://github.com/pallets/flask --k 10 --data-dir data --repos data/repos.json
```

This is useful for quick testing without opening the Streamlit app.

---

## Experimental pipeline in `pt2`

There is also a smaller experiment-oriented pipeline under `pt2/`.

The script `pt2/scripts/run_expanded_pipeline.sh` runs the broader data collection, embedding, and FAISS build path for a curated or expanded repo list.

Example:

```bash
bash pt2/scripts/run_expanded_pipeline.sh
```

This writes generated artifacts into:

- `pt2/datasets_expanded/repos_expanded.json`
- `pt2/datasets_expanded/embeddings.npy`
- `pt2/datasets_expanded/faiss.index`

This folder acts like a “generated-data” workspace for experiments without cluttering the base project with large artifacts.

---

## Why this approach is interesting

The main design idea is that repository similarity should not be reduced to one signal.

A repository is often similar to another one because of:

- shared technical stack
- overlapping dependencies
- similar README language
- related ecosystem or problem domain
- similar project type

The project tries to capture all of those signals together in a single retrieval system, which makes it much more useful for developer navigation and repo discovery.

---

## Example use cases

This project is useful in several situations:

- discovering projects in the same ecosystem or domain
- finding alternative implementations of a library or framework
- exploring similar open-source tools when you already know one good project
- building recommendation features for a developer platform
- evaluating whether semantic and structural signals improve repo retrieval quality

---

## Notes and limitations

This project is a personal prototype, so there are some practical limitations:

- GitHub API rate limits can slow collection if no token is provided
- dependency extraction is heuristic and varies by project structure
- README quality is uneven across repositories
- a large repository corpus requires more compute and storage
- some repos are better matched by domain or intent than by direct technical overlap

Even with those limitations, the project is a strong demonstration of hybrid repo similarity and retrieval design.

---

## Personal project status

This repo represents a hands-on exploration into:

- semantic search
- repository recommendation
- vector retrieval with FAISS
- multi-signal ranking
- developer tooling and open-source discovery

It was built as a practical experiment in turning repository information into a useful similarity engine, while also exploring how AI and retrieval models can help programmers discover related projects more meaningfully.

If you want to extend it further, the next natural steps would be:

- add a richer query-time similarity explanation
- improve dependency normalization across ecosystems
- include commit history or contributor overlap as a signal
- add a lightweight web dashboard or better result filtering
- benchmark across a larger curated dataset

---

## Quick start summary

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m src.collect_repo_data --repo facebook/react --out data/repos.json
python -m src.embeddings --repos data/repos.json --out-dir data
python -m src.similarity build --data-dir data --repos data/repos.json
streamlit run app.py
```
