"""
Streamlit UI: enter a GitHub repo and see ranked similar repositories.
Run from project root: streamlit run app.py
"""

import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
from src.similarity import get_similar_any_repo, load_index

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "pt2" / "datasets_expanded"
REPOS_PATH = PROJECT_ROOT / "pt2" / "datasets_expanded" / "repos_expanded.json"
K_DEFAULT = 10


@st.cache_resource
def load_similarity_artifacts():
    if not (DATA_DIR / "faiss.index").exists():
        return None, None, None, (
            f"FAISS index not found. Run:\n"
            f"python -m src.similarity build --data-dir {DATA_DIR} --repos {REPOS_PATH}"
        )
    index, repos, repo_index = load_index(DATA_DIR, REPOS_PATH)
    repos_map = {r["repo"]: r for r in repos}
    return index, (repos, repo_index), repos_map, None


st.set_page_config(page_title="Repo Similarity", page_icon="🔎", layout="wide")

st.markdown("""
<style>
  .block-container { padding-top: 2rem; max-width: 100rem; }
  details summary p { font-size: 1.0rem; }
  h1 { text-align: center; }
  .explanation-box {
    background: #1a1a2e;
    border-left: 3px solid #6c63ff;
    border-radius: 4px;
    padding: 0.75rem 1rem;
    margin-top: 0.5rem;
    font-size: 0.92rem;
    line-height: 1.5;
    color: #e0e0e0;
  }
  .explanation-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6c63ff;
    font-weight: 600;
    margin-bottom: 0.3rem;
  }
</style>
""", unsafe_allow_html=True)

# Sidebar 
st.sidebar.subheader("🤖 AI Explanations")
groq_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    help="Get a free key at console.groq.com — enables natural language explanations of why repos are similar.",
    placeholder="gsk_...",
)
explain_top_n = st.sidebar.slider(
    "Auto-explain top N results",
    min_value=0,
    max_value=10,
    value=0,
    help="Set to 0 to use the per-result Explain button instead.",
)

st.sidebar.divider()
st.sidebar.subheader("Query options")
k = st.sidebar.slider("Top‑K results", min_value=1, max_value=10, value=K_DEFAULT)
show_examples = st.sidebar.checkbox("Show example indexed repos", value=False)

# Load artifacts 
index, data_tuple, repos_map, load_err = load_similarity_artifacts()
if load_err:
    st.error(load_err)
    st.stop()
repos, repo_index = data_tuple

st.sidebar.caption(f"Index: **{len(repo_index)}** repos")

if show_examples:
    with st.sidebar.expander("Browse indexed repos", expanded=False):
        st.caption("Copy any repo name below to use as a query.")
        sample = (repo_index or [])[:400]
        st.text_area("Indexed repos", value="\n".join(sample), height=200)
        st.download_button(
            "Download full list",
            data=json.dumps(repo_index, indent=2),
            file_name="repo_index.json",
            mime="application/json",
        )

# Main UI 
st.title("Multi‑Signal Repository Discovery")

st.markdown("""
**How it works:** combines multiple signals to find functionally similar repositories:
- **Semantic text similarity** — Sentence-BERT embeddings of README + description
- **Dependency overlap** — Jaccard similarity on package manifests
- **Topic overlap** — GitHub topic tags with keyword fallback for untagged repos
- **Ecosystem & language match** — lightweight structural signals
""".strip())

st.divider()

query = st.text_input(
    "Repository (owner/name or GitHub URL)",
    value="",
    placeholder="e.g. karpathy/micrograd or https://github.com/owner/repo",
    help="Any public GitHub repo. If not in the index, we fetch it live.",
)

# Search 
if st.button("Find similar", type="primary") or query:
    query = query.strip()
    if not query:
        st.warning("Enter owner/repo (e.g. facebook/react).")
        st.stop()
    if "/" not in query:
        st.warning("Use format owner/repo (e.g. facebook/react).")
        st.stop()

    with st.spinner("Searching..."):
        results, err = get_similar_any_repo(
            query, k, DATA_DIR,
            index=index, repos=repos, repo_index=repo_index,
            repos_path=REPOS_PATH,
        )
    if err:
        st.error(err)
        st.stop()
    if not results:
        st.info("No results found.")
        st.stop()

    st.success(f"Top {len(results)} repos similar to **{query}**")

    # get query repo metadat for explanation context
    from src.similarity import parse_repo
    query_normalized = parse_repo(query)
    query_meta = repos_map.get(query_normalized, {})

    # uto-explain top N if configured and key present
    auto_explanations = {}
    if explain_top_n > 0 and groq_key:
        with st.spinner(f"Generating AI explanations for top {explain_top_n} results..."):
            try:
                from src.groq_explain import explain_batch
                auto_explanations = explain_batch(
                    query_repo=query_normalized,
                    query_meta=query_meta,
                    results=results,
                    repos_map=repos_map,
                    top_n=explain_top_n,
                    api_key=groq_key,
                )
            except Exception as e:
                st.warning(f"Auto-explain failed: {e}")

    #  Results 
    for i, r in enumerate(results, 1):
        repo_name = r["repo"]
        with st.expander(f"{i}. **{repo_name}**  •  combined **{r['combined_score']:.3f}**"):
            summary = (r.get("summary") or "").strip()
            if summary:
                st.markdown(f"**Summary:** {summary}")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"- **Text:** {r['text_score']:.3f}")
                st.markdown(f"- **Dependencies:** {r['dep_score']:.3f}")
                st.markdown(f"- **Topics:** {r.get('topic_score', 0):.3f}")
            with c2:
                st.markdown(f"- **Ecosystem:** {r.get('ecosystem_score', 0):.3f}")
                st.markdown(f"- **Language match:** {r.get('language_match', 0):.0f}")
                st.markdown(f"- **Combined:** {r['combined_score']:.3f}")

            st.link_button("Open on GitHub", f"https://github.com/{repo_name}")

           
            if repo_name in auto_explanations:
                st.markdown(
                    f'<div class="explanation-label">🤖 Why similar</div>'
                    f'<div class="explanation-box">{auto_explanations[repo_name]}</div>',
                    unsafe_allow_html=True,
                )
            elif groq_key:
              
                btn_key = f"explain_{i}_{repo_name.replace('/', '_')}"
                if st.button("🤖 Explain why similar", key=btn_key):
                    with st.spinner("Generating explanation..."):
                        try:
                            from src.groq_explain import explain_similarity
                            similar_meta = repos_map.get(repo_name, {})
                            explanation = explain_similarity(
                                query_repo=query_normalized,
                                query_meta=query_meta,
                                similar_repo=repo_name,
                                similar_meta=similar_meta,
                                scores=r,
                                api_key=groq_key,
                            )
                            st.markdown(
                                f'<div class="explanation-label">🤖 Why similar</div>'
                                f'<div class="explanation-box">{explanation}</div>',
                                unsafe_allow_html=True,
                            )
                        except Exception as e:
                            st.warning(f"Could not generate explanation: {e}")
            else:
                st.caption("💡 Add your Groq API key in the sidebar to enable AI explanations.")
