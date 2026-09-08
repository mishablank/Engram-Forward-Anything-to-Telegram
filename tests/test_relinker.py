from __future__ import annotations

from pathlib import Path

from engram import relinker as relinker_module
from engram.embeddings import SemanticIndex
from engram.relinker import (
    format_related_line,
    relink_note,
)
from tests.test_embeddings import StubEmbedder


def _write(p: Path, body: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


def _setup_index(tmp_path: Path) -> SemanticIndex:
    emb = StubEmbedder()
    idx = SemanticIndex(tmp_path, emb)
    idx.refresh()
    return idx


def test_format_related_line() -> None:
    assert format_related_line(["a", "b"]) == "*Related: [[a]] · [[b]]*"


def test_relink_inserts_new_block_when_absent(tmp_path: Path) -> None:
    _write(tmp_path / "AI" / "target.md", "---\n---\n\n# Target\n\nrocket launch propulsion\n")
    _write(tmp_path / "AI" / "neighbour.md", "---\n---\n\n# Neighbour\n\nrocket launch propulsion details\n")
    idx = _setup_index(tmp_path)
    target = tmp_path / "AI" / "target.md"

    changed, titles = relink_note(target, idx, min_score=0.0)
    assert changed
    assert "neighbour" in titles
    text = target.read_text(encoding="utf-8")
    assert "*Related: [[neighbour]]*" in text


def test_relink_replaces_existing_block(tmp_path: Path) -> None:
    target = tmp_path / "AI" / "target.md"
    _write(
        target,
        "---\n---\n\n# Target\n\nrocket launch\n\n---\n*Related: [[stale_link]]*\n",
    )
    _write(tmp_path / "AI" / "neighbour.md", "---\n---\n\nrocket launch propulsion details\n")
    idx = _setup_index(tmp_path)

    changed, titles = relink_note(target, idx, min_score=0.0)
    assert changed
    text = target.read_text(encoding="utf-8")
    assert "stale_link" not in text
    assert "neighbour" in text
    # Body content preserved.
    assert "rocket launch" in text


def test_relink_no_change_when_titles_match(tmp_path: Path) -> None:
    target = tmp_path / "AI" / "target.md"
    _write(target, "---\n---\n\nrocket\n\n---\n*Related: [[neighbour]]*\n")
    _write(tmp_path / "AI" / "neighbour.md", "---\n---\n\nrocket details\n")
    idx = _setup_index(tmp_path)

    # Pre-set the existing line to match what relinker will compute.
    changed1, titles1 = relink_note(target, idx, min_score=0.0)
    # Run again — should be a no-op now.
    changed2, titles2 = relink_note(target, idx, min_score=0.0)
    assert changed2 is False
    assert titles1 == titles2


def test_relink_skips_when_index_disabled(tmp_path: Path) -> None:
    target = tmp_path / "AI" / "target.md"
    _write(target, "body\n")
    idx = SemanticIndex(tmp_path, embedder=None)
    idx.refresh()
    changed, titles = relink_note(target, idx)
    assert not changed
    assert titles == []


def test_relink_skips_unavailable_icloud_file(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "AI" / "target.md"
    _write(target, "body\n")
    index = _setup_index(tmp_path)
    monkeypatch.setattr(relinker_module, "is_available", lambda _: False)

    assert relink_note(target, index) == (False, [])


def test_relink_preserves_trailing_tags_line(tmp_path: Path) -> None:
    target = tmp_path / "AI" / "target.md"
    _write(
        target,
        "---\n---\n\n# Target\n\nrocket launch propulsion\n\n#ai #ml\n",
    )
    _write(tmp_path / "AI" / "neighbour.md", "---\n---\n\nrocket launch propulsion details\n")
    idx = _setup_index(tmp_path)
    changed, _ = relink_note(target, idx, min_score=0.0)
    assert changed
    text = target.read_text(encoding="utf-8")
    # Tags line still trails the related block.
    related_pos = text.find("*Related:")
    tags_pos = text.find("#ai")
    assert 0 < related_pos < tags_pos


def test_relink_drops_block_when_no_candidates_clear_threshold(tmp_path: Path) -> None:
    target = tmp_path / "AI" / "target.md"
    _write(
        target,
        "---\n---\n\nrocket\n\n---\n*Related: [[old_link]]*\n",
    )
    # No other notes in the vault → no candidates.
    idx = _setup_index(tmp_path)
    changed, titles = relink_note(target, idx, min_score=0.0)
    assert changed
    assert titles == []
    text = target.read_text(encoding="utf-8")
    assert "Related" not in text
    assert "old_link" not in text
