from __future__ import annotations

import logging
import re
from pathlib import Path

from .embeddings import SemanticIndex
from .vault import is_available

log = logging.getLogger(__name__)

RELATED_LINE_RE = re.compile(r"^\*Related: .*?\*[ \t]*$", re.MULTILINE)
DEFAULT_K = 5
MIN_SCORE = 0.55  # don't pollute notes with weak links


def format_related_line(titles: list[str]) -> str:
    return "*Related: " + " · ".join(f"[[{t}]]" for t in titles) + "*"


def _extract_titles(line: str) -> list[str]:
    return re.findall(r"\[\[([^\]]+)\]\]", line)


def relink_note(
    path: Path,
    semantic_index: SemanticIndex,
    *,
    k: int = DEFAULT_K,
    min_score: float = MIN_SCORE,
) -> tuple[bool, list[str]]:
    """Recompute the `*Related: ...*` line for one note.

    Returns (changed, titles). `changed=False` when the new set matches the existing
    line, or when the note isn't in the semantic index, or when no candidates clear
    `min_score`.
    """
    if not semantic_index.enabled or not is_available(path):
        return False, []
    hits = semantic_index.nearest_for_path(path, k=k)
    new_titles = [h.path.stem for h in hits if h.score >= min_score]
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False, []

    existing_match = RELATED_LINE_RE.search(text)
    existing_titles = _extract_titles(existing_match.group(0)) if existing_match else []

    if new_titles == existing_titles:
        return False, new_titles

    if not new_titles:
        if existing_match is None:
            return False, []
        # Strip the existing related line (and a leading `---` separator if it became
        # orphaned) without disturbing the rest of the note.
        new_text = _strip_related_block(text, existing_match)
        path.write_text(new_text, encoding="utf-8")
        return True, []

    new_line = format_related_line(new_titles)
    if existing_match is not None:
        new_text = text[: existing_match.start()] + new_line + text[existing_match.end():]
    else:
        new_text = _insert_related_block(text, new_line)
    path.write_text(new_text, encoding="utf-8")
    return True, new_titles


def _strip_related_block(text: str, match: re.Match[str]) -> str:
    start = match.start()
    end = match.end()
    # Walk back through whitespace and a leading `---` divider line.
    line_start = text.rfind("\n", 0, start) + 1
    prefix = text[:line_start]
    # If the line above is `---`, drop it too (it was just the separator we wrote).
    above_end = line_start - 1  # the \n we just found
    above_start = text.rfind("\n", 0, above_end) + 1
    if text[above_start:above_end].strip() == "---":
        prefix = text[:above_start]
    # Drop trailing newline after the related line for cleanliness.
    suffix = text[end:]
    if suffix.startswith("\n"):
        suffix = suffix[1:]
    return prefix + suffix


def _insert_related_block(text: str, new_line: str) -> str:
    block = f"\n---\n{new_line}\n"
    # Insert before a trailing tags line (e.g. "#tag1 #tag2") if present.
    tag_match = re.search(r"\n(#\S+(?:\s+#\S+)*)\s*$", text)
    if tag_match is not None:
        return text[: tag_match.start()] + block + text[tag_match.start():]
    if not text.endswith("\n"):
        text += "\n"
    return text + block.lstrip("\n")
