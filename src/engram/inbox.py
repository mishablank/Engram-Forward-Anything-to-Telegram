from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

from .vault import FRONTMATTER_RE, is_available

log = logging.getLogger(__name__)

REVIEW_LINE_RE = re.compile(r"^review\s*:\s*pending\s*$", re.MULTILINE | re.IGNORECASE)
IGNORE_DIRS = ("attachments",)


def _read(path: Path) -> str:
    if not is_available(path):
        log.warning("Skipping unavailable iCloud file: %s", path)
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def is_pending(text: str) -> bool:
    fm = FRONTMATTER_RE.match(text)
    if not fm:
        return False
    return bool(REVIEW_LINE_RE.search(fm.group(1)))


def find_pending(base_dir: Path) -> list[Path]:
    """Return paths of notes whose frontmatter contains `review: pending`,
    sorted by mtime (oldest first)."""
    pending: list[tuple[float, Path]] = []
    for path in base_dir.rglob("*.md"):
        rel_parts = path.relative_to(base_dir).parts
        if any(part in IGNORE_DIRS for part in rel_parts):
            continue
        if rel_parts and rel_parts[0].startswith("."):
            continue
        if not is_available(path):
            log.warning("Skipping unavailable iCloud file: %s", path)
            continue
        text = _read(path)
        if is_pending(text):
            try:
                pending.append((path.stat().st_mtime, path))
            except OSError:
                continue
    pending.sort(key=lambda x: x[0])
    return [p for _, p in pending]


def clear_pending(path: Path) -> bool:
    """Remove the `review: pending` line from a note's frontmatter. Returns True if changed."""
    text = _read(path)
    fm = FRONTMATTER_RE.match(text)
    if not fm:
        return False
    fm_body = fm.group(1)
    new_fm_body, count = REVIEW_LINE_RE.subn("", fm_body)
    if count == 0:
        return False
    # Collapse runs of blank lines created by deletion.
    new_fm_body = re.sub(r"\n{3,}", "\n\n", new_fm_body).strip("\n")
    new_text = f"---\n{new_fm_body}\n---{text[fm.end():]}"
    path.write_text(new_text, encoding="utf-8")
    return True


def move_note(path: Path, target_dir: Path) -> Path:
    """Move a note to target_dir; appends ' (N)' on filename collision."""
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / path.name
    n = 2
    while dest.exists():
        dest = target_dir / f"{path.stem} ({n}){path.suffix}"
        n += 1
    shutil.move(str(path), str(dest))
    return dest


def preview(path: Path, max_lines: int = 6) -> str:
    text = _read(path)
    fm = FRONTMATTER_RE.match(text)
    body = text[fm.end():] if fm else text
    body = body.strip()
    lines = [ln for ln in body.splitlines() if ln.strip()]
    return "\n".join(lines[:max_lines])
