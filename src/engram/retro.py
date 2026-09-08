from __future__ import annotations

import logging
from pathlib import Path

from anthropic import Anthropic

from .dedupe import find_semantic_duplicate
from .embeddings import SemanticIndex
from .merger import merge_note
from .note_writer import is_safe_merge
from .vault import is_available, load_note_body

log = logging.getLogger(__name__)


def merge_duplicate_notes(
    base_dir: Path,
    semantic_index: SemanticIndex | None,
    client: Anthropic,
    paths: list[Path],
) -> int:
    """Collapse semantically-duplicate notes across the vault into canonical pages.

    For each note, find its best duplicate among the others (cosine + LLM judge,
    same thresholds as live capture). When found, merge this note's body into the
    canonical match and delete this note. Returns the number of merges performed.

    Destructive: callers must `gitsafe.snapshot` first.
    """
    if semantic_index is None or not semantic_index.enabled:
        return 0
    deleted: set[Path] = set()
    merges = 0
    for note in paths:
        if note in deleted or not note.exists() or not is_available(note):
            continue
        body = load_note_body(note)
        if not body.strip():
            continue
        target = find_semantic_duplicate(
            semantic_index,
            client,
            note.stem,
            body,
            skip_paths=deleted | {note},
        )
        if (
            target is None
            or target in deleted
            or not target.exists()
            or not is_available(target)
        ):
            continue
        # Keep the older note as canonical so its creation date and title
        # survive; merge the newer one into it and delete the newer.
        if note.stat().st_mtime <= target.stat().st_mtime:
            canonical, victim = note, target
        else:
            canonical, victim = target, note
        if not is_available(canonical) or not is_available(victim):
            continue
        existing = canonical.read_text(encoding="utf-8")
        merged = merge_note(client, existing, load_note_body(victim))
        if not is_safe_merge(existing, merged):
            log.info(
                "Retro merge unsafe: %s -> %s; skipping", victim.name, canonical.name
            )
            continue
        canonical.write_text(merged.rstrip() + "\n", encoding="utf-8")
        victim.unlink()
        deleted.add(victim)
        merges += 1
        log.info("Retro merged %s into %s", victim.name, canonical.name)
    return merges
