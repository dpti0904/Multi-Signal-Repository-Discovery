"""
Week 1: Data collection for one or more GitHub repositories.
Fetches README, description, topics, dependencies (package.json, requirements.txt, pyproject.toml,
go.mod, Cargo.toml, Gemfile); optionally stargazers.

Usage:
  python -m src.collect_repo_data --repo facebook/react
  python -m src.collect_repo_data --list data/repo_list.txt --out data/repos.json --resume --save-every 50

Requires: GITHUB_TOKEN in environment for higher rate limits.
"""

import argparse
import base64
import json
import os
import re
import time
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GITHUB_API = "https://api.github.com"

def get_headers(include_topics: bool = False):
    token = os.environ.get("GITHUB_TOKEN")
    accept = "application/vnd.github.v3+json"
    if include_topics:
        accept = "application/vnd.github.mercy-preview+json"
    headers = {"Accept": accept}
    if token:
        headers["Authorization"] = f"token {token}"
    return headers

def fetch_file(owner: str, repo: str, path: str) -> str | None:
    """Get file content from repo. Returns None if not found or error."""
    url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}"
    r = requests.get(url, headers=get_headers())
    if r.status_code != 200:
        return None
    data = r.json()
    if data.get("encoding") == "base64":
        return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    return data.get("content")

def fetch_readme(owner: str, repo: str) -> str:
    """Fetch README content (tries README.md, README, etc. via the readme endpoint)."""
    url = f"{GITHUB_API}/repos/{owner}/{repo}/readme"
    r = requests.get(url, headers=get_headers())
    if r.status_code != 200:
        return ""
    data = r.json()
    if data.get("encoding") == "base64":
        return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    return ""

def parse_package_json(content: str) -> set[str]:
    """Extract dependency names from package.json."""
    deps = set()
    try:
        data = json.loads(content)
        for key in ("dependencies", "devDependencies", "peerDependencies"):
            if key in data and isinstance(data[key], dict):
                deps.update(data[key].keys())
    except json.JSONDecodeError:
        pass
    return deps

def parse_requirements_txt(content: str) -> set[str]:
    """Extract package names from requirements.txt (ignores version specifiers)."""
    deps = set()
    for line in content.splitlines():
        line = line.strip().split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        name = re.split(r"[\s=<>!~]", line)[0].strip()
        if name:
            deps.add(name)
    return deps

def parse_pyproject_toml(content: str) -> set[str]:
    """
    Extract dependencies from pyproject.toml.

    Supports:
    - PEP 621: [project] dependencies = ["pkg>=1", ...]
    - Poetry:  [tool.poetry.dependencies] pkg = "^1.2"
    """
    deps: set[str] = set()

    in_project = False
    capturing = False
    buf: list[str] = []
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line.strip("[]").strip()
            in_project = section == "project"
            capturing = False
            buf.clear()
            continue

        if in_project:
            if not capturing and re.match(r"^dependencies\s*=", line):
                capturing = True
                _, rhs = line.split("=", 1)
                buf.append(rhs.strip())
                if "]" in rhs:
                    capturing = False
                    joined = " ".join(buf)
                    for m in re.findall(r"['\"]([^'\"]+)['\"]", joined):
                        name = re.split(r"[=<>!~\\s\\[]", m.strip())[0].strip()
                        if name and name.lower() != "python":
                            deps.add(name)
                    buf.clear()
            elif capturing:
                buf.append(line)
                if "]" in line:
                    capturing = False
                    joined = " ".join(buf)
                    for m in re.findall(r"['\"]([^'\"]+)['\"]", joined):
                        name = re.split(r"[=<>!~\\s\\[]", m.strip())[0].strip()
                        if name and name.lower() != "python":
                            deps.add(name)
                    buf.clear()

    in_poetry_deps = False
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line.strip("[]").strip()
            in_poetry_deps = section == "tool.poetry.dependencies"
            continue
        if in_poetry_deps and "=" in line:
            name = re.split(r"[\s=<>!~\[\]]", line)[0].strip().strip('"').strip("'")
            if name and name.lower() != "python":
                deps.add(name)

    return deps

def parse_go_mod(content: str) -> set[str]:
    """Extract module names from go.mod require block."""
    deps = set()
    in_require = False
    for line in content.splitlines():
        line = line.strip()
        if line == "require (":
            in_require = True
            continue
        if in_require and line == ")":
            break
        if in_require and line and not line.startswith("//"):
            parts = line.split()
            if parts:
                deps.add(parts[0])
    return deps

def parse_cargo_toml(content: str) -> set[str]:
    """Extract dependencies from Cargo.toml [dependencies]."""
    deps = set()
    in_deps = False
    for line in content.splitlines():
        line = line.strip()
        if line == "[dependencies]" or line.startswith("[dependencies."):
            in_deps = True
            continue
        if in_deps and line.startswith("["):
            break
        if in_deps and "=" in line and not line.startswith("#"):
            name = line.split("=")[0].strip()
            if name:
                deps.add(name)
    return deps

def parse_gemfile(content: str) -> set[str]:
    """Extract gem names from Gemfile."""
    deps = set()
    for line in content.splitlines():
        line = line.strip().split("#")[0].strip()
        if line.startswith("gem ") or line.startswith('gem "'):
            m = re.search(r'gem\s+["\']([^"\']+)["\']', line)
            if m:
                deps.add(m.group(1))
    return deps

def parse_setup_py(content: str) -> set[str]:
    """Best-effort extraction of install_requires-style dependencies from setup.py."""
    deps = set()
    in_install_requires = False
    for line in content.splitlines():
        stripped = line.strip()
        if "install_requires" in stripped and "[" in stripped:
            in_install_requires = True
        if in_install_requires:
            if "]" in stripped:
                in_install_requires = False
            m = re.search(r"['\"]([^'\"]+)['\"]", stripped)
            if m:
                name = re.split(r"[=<>!~]", m.group(1))[0].strip()
                if name and not name.startswith("python"):
                    deps.add(name)
    return deps

def parse_pom_xml(content: str) -> set[str]:
    """Extract Maven artifactIds from pom.xml."""
    deps = set()
    for line in content.splitlines():
        line = line.strip()
        if "<artifactId>" in line:
            m = re.search(r"<artifactId>([^<]+)</artifactId>", line)
            if m:
                deps.add(m.group(1).strip())
    return deps

def parse_build_gradle(content: str) -> set[str]:
    """Extract dependency coordinates from Gradle build files (very roughly)."""
    deps = set()
    for line in content.splitlines():
        line = line.strip().split("//")[0].strip()
        if not line:
            continue
        m = re.search(r"['\"]([^:'\"]+:[^:'\"]+)", line)
        if m:
            coord = m.group(1)
            name = coord.split(":")[1]
            if name:
                deps.add(name)
    return deps

def parse_makefile(content: str) -> set[str]:
    """Very rough: capture -l<lib> flags as dependencies."""
    deps = set()
    for line in content.splitlines():
        for match in re.findall(r"-l([A-Za-z0-9_+-]+)", line):
            deps.add(match)
    return deps

def parse_cmakelists(content: str) -> set[str]:
    """Very rough: capture library names in target_link_libraries calls."""
    deps = set()
    for line in content.splitlines():
        line = line.strip()
        if line.lower().startswith("target_link_libraries"):
            inside = line[line.find("(") + 1 : line.rfind(")")]
            parts = inside.split()
            for lib in parts[1:]:
                deps.add(lib.strip())
    return deps

MANIFESTS = [
    ("package.json", parse_package_json, "node"),
    ("requirements.txt", parse_requirements_txt, "python"),
    ("pyproject.toml", parse_pyproject_toml, "python"),
    ("setup.py", parse_setup_py, "python"),
    ("pom.xml", parse_pom_xml, "java"),
    ("build.gradle", parse_build_gradle, "java"),
    ("Makefile", parse_makefile, "c-cpp"),
    ("CMakeLists.txt", parse_cmakelists, "c-cpp"),
    ("go.mod", parse_go_mod, "go"),
    ("Cargo.toml", parse_cargo_toml, "rust"),
    ("Gemfile", parse_gemfile, "ruby"),
]

_MONOREPO_DIRS = {"packages", "apps", "services", "frontend", "backend", "src"}
_MONOREPO_FILES = {"lerna.json", "pnpm-workspace.yaml", "nx.json", "rush.json"}

def get_dependencies(owner: str, repo: str) -> tuple[set[str], set[str]]:
    """
    Fast two-phase dependency fetch.

    Phase 1: fetch root directory listing once (1 API call), check which
    manifests exist there, fetch only those files.

    Phase 2: only scan subdirectories if root has zero manifests AND the repo
    looks like a monorepo (has packages/, apps/, lerna.json, etc.). Even then,
    only checks known monorepo subdirectory names — not a full recursive walk.

    Typical cost: 1 listing call + 1-3 manifest fetches = ~2-4 API calls/repo
    vs. the old approach which could make 100+ calls per repo.
    """
    all_deps: set[str] = set()
    ecosystems: set[str] = set()
    manifest_names = {name for name, _, _ in MANIFESTS}

    url = f"{GITHUB_API}/repos/{owner}/{repo}/contents"
    r = requests.get(url, headers=get_headers())
    if r.status_code != 200:
        return all_deps, ecosystems

    try:
        entries = r.json()
    except Exception:
        return all_deps, ecosystems

    if not isinstance(entries, list):
        return all_deps, ecosystems

    root_files = {e["name"] for e in entries if e.get("type") == "file"}
    root_dirs = {e["name"] for e in entries if e.get("type") == "dir"}

    found_manifests = root_files & manifest_names

    for name in found_manifests:
        entry = next((m for m in MANIFESTS if m[0] == name), None)
        if not entry:
            continue
        _, parser, ecosystem = entry
        content = fetch_file(owner, repo, name)
        if content:
            all_deps |= parser(content)
            ecosystems.add(ecosystem)

    if not found_manifests:
        is_monorepo = bool(
            (root_dirs & _MONOREPO_DIRS) or (root_files & _MONOREPO_FILES)
        )
        if is_monorepo:
            for dirname in sorted(root_dirs & _MONOREPO_DIRS):
                suburl = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{dirname}"
                sr = requests.get(suburl, headers=get_headers())
                if sr.status_code != 200:
                    continue
                try:
                    sub_entries = sr.json()
                except Exception:
                    continue
                if not isinstance(sub_entries, list):
                    continue
                for sub in sub_entries:
                    if sub.get("type") != "dir":
                        continue
                    sub_name = sub.get("name", "")
                    for mname, parser, ecosystem in MANIFESTS:
                        content = fetch_file(owner, repo, f"{dirname}/{sub_name}/{mname}")
                        if content:
                            all_deps |= parser(content)
                            ecosystems.add(ecosystem)

    return all_deps, ecosystems

def fetch_repo_metadata(owner: str, repo: str) -> dict:
    """Fetch core repo metadata from the repos API (topics, language, stars, size, etc.)."""
    url = f"{GITHUB_API}/repos/{owner}/{repo}"
    r = requests.get(url, headers=get_headers(include_topics=True))
    if r.status_code != 200:
        return {}
    data = r.json()
    return {
        "description": data.get("description") or "",
        "topics": data.get("topics") or [],
        "language": data.get("language") or "",
        "stargazers_count": data.get("stargazers_count", 0),
        "forks_count": data.get("forks_count", 0),
        "watchers_count": data.get("watchers_count", 0),
        "created_at": data.get("created_at"),
        "updated_at": data.get("updated_at"),
        "pushed_at": data.get("pushed_at"),
        "is_fork": bool(data.get("fork")),
        "license": (data.get("license") or {}).get("spdx_id"),
        "size": data.get("size", 0),
    }

def fetch_languages(owner: str, repo: str) -> dict:
    """Fetch language breakdown (bytes of code per language)."""
    url = f"{GITHUB_API}/repos/{owner}/{repo}/languages"
    r = requests.get(url, headers=get_headers())
    if r.status_code != 200:
        return {}
    try:
        data = r.json()
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}

def infer_project_type(repo: str, description: str, topics: list[str], readme: str) -> str:
    """
    Heuristic project type classifier.
    Types: algorithms, tutorial, exercises, awesome-list, style-guide, library.
    """
    text = " ".join([
        repo or "",
        description or "",
        " ".join(topics or []),
        (readme or "")[:4000],
    ]).lower()

    if "awesome " in text or "awesome-" in text or "curated list" in text:
        return "awesome-list"
    if "style guide" in text or "styleguide" in text or "best practices" in text:
        return "style-guide"
    if "exercise" in text or "exercises" in text or "practice" in text:
        return "exercises"
    if "algorithm" in text or "data structure" in text or "data-structures" in text:
        return "algorithms"
    if "tutorial" in text or "guide" in text or "30 days" in text or "30-day" in text:
        return "tutorial"
    return "library"

def fetch_stargazers(owner: str, repo: str, max_pages: int = 2) -> list[str]:
    """Fetch stargazer logins (for collaborative filtering). Rate limited: 30/min without token."""
    url = f"{GITHUB_API}/repos/{owner}/{repo}/stargazers"
    logins = []
    page = 1
    while page <= max_pages:
        r = requests.get(url, headers=get_headers(), params={"per_page": 100, "page": page})
        if r.status_code != 200:
            break
        data = r.json()
        if not data:
            break
        logins.extend([u["login"] for u in data])
        page += 1
        time.sleep(0.5)
    return logins

def collect_one(owner: str, repo: str, include_stars: bool = False) -> dict:
    """Collect README, metadata (description, topics, language), dependencies, ecosystems, and optionally stargazers."""
    full_name = f"{owner}/{repo}"
    readme = fetch_readme(owner, repo)

    cleaned_readme_lines: list[str] = []
    for i, line in enumerate(readme.splitlines()):
        stripped = line.strip()
        if i < 20 and (
            ("shields.io" in stripped)
            or stripped.startswith("[![")
            or stripped.startswith("<img")
        ):
            continue
        cleaned_readme_lines.append(line)
    readme = "\n".join(cleaned_readme_lines)

    deps, ecosystems = get_dependencies(owner, repo)
    meta = fetch_repo_metadata(owner, repo)
    languages_breakdown = fetch_languages(owner, repo)
    project_type = infer_project_type(
        full_name,
        meta.get("description", ""),
        meta.get("topics", []),
        readme,
    )
    out = {
        "repo": full_name,
        "owner": owner,
        "name": repo,
        "readme_text": readme,
        "description": meta.get("description", ""),
        "topics": meta.get("topics", []),
        "language": meta.get("language", ""),
        "dependencies": list(deps),
        "ecosystems": list(ecosystems),
        "stargazers_count": meta.get("stargazers_count", 0),
        "forks_count": meta.get("forks_count", 0),
        "watchers_count": meta.get("watchers_count", 0),
        "created_at": meta.get("created_at"),
        "updated_at": meta.get("updated_at"),
        "pushed_at": meta.get("pushed_at"),
        "is_fork": meta.get("is_fork", False),
        "license": meta.get("license"),
        "size": meta.get("size", 0),
        "languages": languages_breakdown,
        "project_type": project_type,
    }
    if include_stars:
        out["stargazers"] = fetch_stargazers(owner, repo)
    return out

def main():
    parser = argparse.ArgumentParser(description="Collect repo data for multi-signal similarity.")
    parser.add_argument("--repo", type=str, help="Single repo as owner/name, e.g. facebook/react")
    parser.add_argument("--list", type=str, help="Path to file with one owner/name per line")
    parser.add_argument("--stars", action="store_true", help="Fetch stargazers (rate limited)")
    parser.add_argument("--out", type=str, default="data/repos.json", help="Output path")
    parser.add_argument("--resume", action="store_true", help="Skip repos already in --out (append to existing)")
    parser.add_argument("--save-every", type=int, default=0, help="Save progress every N repos (0 = only at end)")
    args = parser.parse_args()

    repos = []
    if args.repo:
        if "/" not in args.repo:
            print("Use owner/name, e.g. facebook/react")
            return
        owner, name = args.repo.split("/", 1)
        repos.append((owner.strip(), name.strip()))
    elif args.list:
        path = Path(args.list)
        if not path.exists():
            print(f"File not found: {path}")
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip().split("#")[0].strip()
            if "/" in line:
                owner, name = line.split("/", 1)
                repos.append((owner.strip(), name.strip()))
    else:
        parser.print_help()
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    done_repos = set()
    if args.resume and out_path.exists():
        try:
            with open(out_path, encoding="utf-8") as f:
                results = json.load(f)
            done_repos = {r["repo"] for r in results}
            print(f"Resuming: {len(results)} repos already in {out_path}")
        except (json.JSONDecodeError, KeyError):
            pass

    todo = [(o, n) for o, n in repos if f"{o}/{n}" not in done_repos]
    if not todo:
        print("Nothing to do (all repos already collected).")
        return

    collected_this_run = 0
    for i, (owner, name) in enumerate(todo):
        print(f"[{len(results) + 1}/{len(repos)}] {owner}/{name}")
        try:
            row = collect_one(owner, name, include_stars=args.stars)
            results.append(row)
            collected_this_run += 1
        except Exception as e:
            print(f"  Error: {e}")
        time.sleep(0.4)

        if args.save_every and collected_this_run > 0 and collected_this_run % args.save_every == 0:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  Saved {len(results)} repos")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(results)} repos to {out_path}")

if __name__ == "__main__":
    main()