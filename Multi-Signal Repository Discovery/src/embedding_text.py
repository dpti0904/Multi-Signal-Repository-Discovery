"""
Shared logic for building the text we embed (README + description + topics).
Used by embeddings.py when building the index and by similarity.py when
embedding an out-of-index query repo. Keeps query and index comparable.
"""

def build_embedding_text(repo: dict, readme_max_chars: int = 3000) -> str:
    """
    Build text for embedding: description + topics + first N chars of README.
    Puts the most discriminative content (description, topics) first; README
    cap keeps it fast and focuses on the top of the README.
    """
    parts = []
    if repo.get("description"):
        parts.append(repo["description"])
    if repo.get("topics"):
        parts.append(" ".join(repo["topics"]))
    readme = (repo.get("readme_text") or "").strip()
    if readme:
        parts.append(readme[:readme_max_chars])
    if not parts:
        return "No README"
    return " ".join(parts)
