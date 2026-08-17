# -*- coding: utf-8 -*-
"""
LLM-powered explanations for repository similarity using Groq.

Given a query repo and a similar repo with their similarity signals,
generates a natural language explanation of WHY they are similar.

Usage:
  from src.groq_explain import explain_similarity

  explanation = explain_similarity(
      query_repo="karpathy/micrograd",
      query_meta={...},
      similar_repo="geohot/tinygrad",
      similar_meta={...},
      scores={
          "text_score": 0.72,
          "dep_score": 0.43,
          "topic_score": 0.38,
          "ecosystem_score": 1.0,
          "language_match": 1,
          "combined_score": 0.61,
      }
  )
"""

import os
import re
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _shared_deps(deps_a: list, deps_b: list, max_show: int = 5) -> list:
    """Return shared dependency names (normalized, limited to max_show)."""
    set_a = {str(x).strip().lower() for x in (deps_a or []) if x}
    set_b = {str(x).strip().lower() for x in (deps_b or []) if x}
    shared = sorted(set_a & set_b)
    return shared[:max_show]


def _shared_topics(topics_a: list, topics_b: list, max_show: int = 5) -> list:
    """Return shared GitHub topics."""
    set_a = set(topics_a or [])
    set_b = set(topics_b or [])
    return sorted(set_a & set_b)[:max_show]


def _build_prompt(
    query_repo: str,
    query_meta: dict,
    similar_repo: str,
    similar_meta: dict,
    scores: dict,
) -> str:
    """Build the prompt for Groq to explain why two repos are similar."""

    shared_deps = _shared_deps(
        query_meta.get("dependencies", []),
        similar_meta.get("dependencies", []),
    )
    shared_topics = _shared_topics(
        query_meta.get("topics", []),
        similar_meta.get("topics", []),
    )

    q_lang = query_meta.get("language") or "unknown"
    s_lang = similar_meta.get("language") or "unknown"
    q_desc = (query_meta.get("description") or "").strip()[:200]
    s_desc = (similar_meta.get("description") or "").strip()[:200]
    q_type = query_meta.get("project_type") or ""
    s_type = similar_meta.get("project_type") or ""

    dep_pct = int(scores.get("dep_score", 0) * 100)
    text_score = scores.get("text_score", 0)
    topic_score = scores.get("topic_score", 0)
    combined = scores.get("combined_score", 0)

    # Determine dominant signal for framing
    signals = {
        "semantic similarity": text_score * 0.45,
        "shared dependencies": scores.get("dep_score", 0) * 0.25,
        "topic overlap": topic_score * 0.15,
    }
    dominant_signal = max(signals, key=signals.get)

    shared_deps_str = (
        "including: " + ", ".join(shared_deps) if shared_deps else "none found"
    )
    shared_topics_str = (
        "shared tags: " + ", ".join(shared_topics) if shared_topics else "none"
    )

    prompt = (
        "You are an expert software engineer explaining why two GitHub repositories are similar.\n"
        "Be concise, specific, and technically accurate. Write 2-3 sentences maximum.\n"
        "Focus on what makes them functionally or technically related, not just that they scored similarly.\n"
        "Do not mention similarity scores, metrics, or how the comparison was made.\n\n"
        f"Query repository: {query_repo}\n"
        f"- Description: {q_desc or 'No description'}\n"
        f"- Language: {q_lang}\n"
        f"- Type: {q_type}\n\n"
        f"Similar repository: {similar_repo}\n"
        f"- Description: {s_desc or 'No description'}\n"
        f"- Language: {s_lang}\n"
        f"- Type: {s_type}\n\n"
        f"Similarity signals:\n"
        f"- Semantic text similarity: {text_score:.2f}/1.0\n"
        f"- Shared dependencies: {dep_pct}% overlap ({shared_deps_str})\n"
        f"- Topic overlap: {topic_score:.2f}/1.0 ({shared_topics_str})\n"
        f"- Same language: {'yes' if scores.get('language_match') else 'no'}\n"
        f"- Combined similarity score: {combined:.3f}/1.0\n"
        f"- Dominant signal: {dominant_signal}\n\n"
        f"Write a 2-3 sentence explanation of why {query_repo} and {similar_repo} are similar.\n"
        "Be specific about what they share technically.\n"
        "Start directly with the technical explanation."
    )

    return prompt


def explain_similarity(
    query_repo: str,
    query_meta: dict,
    similar_repo: str,
    similar_meta: dict,
    scores: dict,
    model: str = "llama-3.1-8b-instant",
    max_tokens: int = 150,
    api_key: Optional[str] = None,
) -> str:
    """
    Generate a natural language explanation of why two repos are similar.

    Returns explanation string, or an error message string if Groq is unavailable.
    Never raises -- safe to call in a Streamlit app.
    """
    key = api_key or os.environ.get("GROQ_API_KEY", "")
    if not key:
        return "Set GROQ_API_KEY in your .env file to enable explanations."

    try:
        from groq import Groq
    except ImportError:
        return "Run: pip install groq"

    prompt = _build_prompt(query_repo, query_meta, similar_repo, similar_meta, scores)

    try:
        client = Groq(api_key=key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        text = response.choices[0].message.content.strip()
        # Clean up any leading artifacts
        text = re.sub(r'^(explanation:|note:|answer:)\s*', '', text, flags=re.IGNORECASE)
        return text
    except Exception as e:
        return f"Groq error: {str(e)[:120]}"


def explain_batch(
    query_repo: str,
    query_meta: dict,
    results: list,
    repos_map: dict,
    top_n: int = 3,
    **kwargs,
) -> dict:
    """
    Generate explanations for the top N results.
    Returns {repo_name: explanation_string}.
    """
    explanations = {}
    for r in results[:top_n]:
        repo_name = r.get("repo", "")
        similar_meta = repos_map.get(repo_name, {})
        explanation = explain_similarity(
            query_repo=query_repo,
            query_meta=query_meta,
            similar_repo=repo_name,
            similar_meta=similar_meta,
            scores=r,
            **kwargs,
        )
        explanations[repo_name] = explanation
    return explanations