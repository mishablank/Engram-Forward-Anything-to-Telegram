from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
import logging
from pathlib import Path

TAG_RE = re.compile(r"(?:^|\s)#([A-Za-z][\w/-]*)")
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)
URL_IN_TEXT_RE = re.compile(r"https?://[^\s\)\]\"'<>]+")
NORM_PUNCT = re.compile(r"[^\w\s]")

SNIPPET_WINDOW = 80
DEFAULT_SEARCH_K = 10
MAX_BODY_CHARS_FOR_RAG = 4000
SF_DATALESS = 0x40000000

log = logging.getLogger("engram")


@dataclass
class SearchHit:
    path: Path
    title: str
    snippet: str
    score: int
    category: str  # vault-relative folder, e.g. "AI"


@dataclass
class VaultIndex:
    titles: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    url_to_path: dict[str, Path] = field(default_factory=dict)
    norm_title_to_path: dict[str, Path] = field(default_factory=dict)

    def find_by_url(self, url: str) -> Path | None:
        return self.url_to_path.get(url.rstrip("/"))

    def find_by_title(self, title: str) -> Path | None:
        return self.norm_title_to_path.get(_normalize_title(title))


def _normalize_title(title: str) -> str:
    title = NORM_PUNCT.sub(" ", title.lower())
    return re.sub(r"\s+", " ", title).strip()


def _is_dataless(stat_result: object) -> bool:
    """Whether macOS has evicted this file's contents from local storage."""
    return bool(getattr(stat_result, "st_flags", 0) & SF_DATALESS)


def is_available(path: Path) -> bool:
    """Whether a note can be read without asking iCloud to download it."""
    try:
        return not _is_dataless(path.stat())
    except OSError:
        return False


def _extract_frontmatter_fields(text: str) -> tuple[set[str], set[str]]:
    """Returns (tags, urls) from frontmatter."""
    tags: set[str] = set()
    urls: set[str] = set()
    fm_match = FRONTMATTER_RE.match(text)
    if not fm_match:
        return tags, urls
    fm = fm_match.group(1)
    current_key: str | None = None
    for raw_line in fm.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            current_key = None
            continue
        if stripped.startswith("- "):
            value = stripped[2:].strip().strip("'\"")
            if not value:
                continue
            if current_key == "tags":
                tags.add(value)
            elif current_key == "urls":
                urls.add(value.rstrip("/"))
            continue
        if ":" in stripped:
            key, _, rest = stripped.partition(":")
            key = key.strip().lower()
            rest = rest.strip()
            if key == "tags":
                if rest.startswith("[") and rest.endswith("]"):
                    for t in rest[1:-1].split(","):
                        t = t.strip().strip("'\"")
                        if t:
                            tags.add(t)
                    current_key = None
                elif rest:
                    tags.add(rest.strip("'\""))
                    current_key = None
                else:
                    current_key = "tags"
            elif key == "urls":
                if rest.startswith("[") and rest.endswith("]"):
                    for u in rest[1:-1].split(","):
                        u = u.strip().strip("'\"")
                        if u:
                            urls.add(u.rstrip("/"))
                    current_key = None
                elif rest:
                    urls.add(rest.strip("'\"").rstrip("/"))
                    current_key = None
                else:
                    current_key = "urls"
            elif key == "url":
                if rest:
                    urls.add(rest.strip("'\"").rstrip("/"))
                current_key = None
            else:
                current_key = None
    return tags, urls


def _extract_tags(text: str) -> list[str]:
    tags, _ = _extract_frontmatter_fields(text)
    fm_match = FRONTMATTER_RE.match(text)
    body = text[fm_match.end():] if fm_match else text
    for m in TAG_RE.finditer(body):
        tags.add(m.group(1))
    return sorted(tags)


def _extract_urls(text: str) -> set[str]:
    """All URLs: from frontmatter and from body (e.g. `> Source:` lines)."""
    _, urls = _extract_frontmatter_fields(text)
    fm_match = FRONTMATTER_RE.match(text)
    body = text[fm_match.end():] if fm_match else text
    for m in URL_IN_TEXT_RE.findall(body):
        urls.add(m.rstrip(".,;:!?").rstrip("/"))
    return urls


def scan_vault(root: Path, ignore_dirs: tuple[str, ...] = ("attachments",)) -> VaultIndex:
    titles: list[str] = []
    tag_counter: Counter[str] = Counter()
    url_to_path: dict[str, Path] = {}
    norm_title_to_path: dict[str, Path] = {}

    paths: list[tuple[float, Path]] = []
    for path in root.rglob("*.md"):
        if any(part in ignore_dirs for part in path.relative_to(root).parts):
            continue
        try:
            stat_result = path.stat()
            if _is_dataless(stat_result):
                log.warning("Skipping unavailable iCloud file: %s", path)
                continue
        except OSError:
            continue
        paths.append((stat_result.st_mtime, path))

    for _, path in sorted(paths):
        titles.append(path.stem)
        norm_title_to_path[_normalize_title(path.stem)] = path
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for tag in _extract_tags(text):
            tag_counter[tag] += 1
        for url in _extract_urls(text):
            url_to_path[url] = path  # later (more recent) wins

    titles.sort()
    top_tags = [t for t, _ in tag_counter.most_common(200)]
    return VaultIndex(
        titles=titles,
        tags=top_tags,
        url_to_path=url_to_path,
        norm_title_to_path=norm_title_to_path,
    )


def _strip_frontmatter(text: str) -> str:
    fm_match = FRONTMATTER_RE.match(text)
    return text[fm_match.end():] if fm_match else text


def _make_snippet(body: str, terms: list[str]) -> str:
    body_lower = body.lower()
    pos = -1
    for t in terms:
        i = body_lower.find(t)
        if i >= 0 and (pos < 0 or i < pos):
            pos = i
    if pos < 0:
        clean = " ".join(body.split())
        return clean[: SNIPPET_WINDOW * 2]
    start = max(0, pos - SNIPPET_WINDOW)
    end = min(len(body), pos + SNIPPET_WINDOW)
    snippet = body[start:end]
    snippet = " ".join(snippet.split())
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(body) else ""
    return f"{prefix}{snippet}{suffix}"


def search_vault(
    root: Path,
    query: str,
    k: int = DEFAULT_SEARCH_K,
    ignore_dirs: tuple[str, ...] = ("attachments",),
) -> list[SearchHit]:
    terms = [t.lower() for t in re.split(r"\s+", query.strip()) if t.strip()]
    if not terms:
        return []
    hits: list[SearchHit] = []
    for path in root.rglob("*.md"):
        rel_parts = path.relative_to(root).parts
        if any(part in ignore_dirs for part in rel_parts):
            continue
        try:
            if _is_dataless(path.stat()):
                log.warning("Skipping unavailable iCloud file: %s", path)
                continue
        except OSError:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        title = path.stem
        title_lower = title.lower()
        body = _strip_frontmatter(text)
        body_lower = body.lower()
        tags_in_note = set(_extract_tags(text))
        score = 0
        for t in terms:
            if t in title_lower:
                score += 10
            if t in tags_in_note:
                score += 5
            score += body_lower.count(t)
        if score == 0:
            continue
        category = path.parent.relative_to(root).as_posix() if path.parent != root else "/"
        hits.append(
            SearchHit(
                path=path,
                title=title,
                snippet=_make_snippet(body, terms),
                score=score,
                category=category,
            )
        )
    hits.sort(key=lambda h: (-h.score, h.title.lower()))
    return hits[:k]


def load_note_body(path: Path, max_chars: int = MAX_BODY_CHARS_FOR_RAG) -> str:
    if not is_available(path):
        log.warning("Skipping unavailable iCloud file: %s", path)
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    body = _strip_frontmatter(text).strip()
    if len(body) > max_chars:
        return body[:max_chars] + "\n…[truncated]"
    return body
