from __future__ import annotations

from pathlib import Path

from engram import inbox as inbox_module
from engram.inbox import (
    clear_pending,
    find_pending,
    is_pending,
    move_note,
    preview,
)


def _write(p: Path, body: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


def test_is_pending_detects_review_field() -> None:
    text = "---\ncreated: 2026-05-11T10:00:00\nreview: pending\n---\n\nbody\n"
    assert is_pending(text)


def test_is_pending_false_when_field_absent() -> None:
    text = "---\ncreated: 2026-05-11T10:00:00\n---\n\nbody\n"
    assert not is_pending(text)


def test_find_pending_returns_oldest_first(tmp_path: Path) -> None:
    a = tmp_path / "AI" / "a.md"
    b = tmp_path / "Other" / "b.md"
    c = tmp_path / "AI" / "c.md"
    _write(a, "---\nreview: pending\n---\n\nA\n")
    _write(b, "---\nreview: pending\n---\n\nB\n")
    _write(c, "---\n---\n\nC\n")

    import os
    os.utime(a, (1_000_000_000, 1_000_000_000))
    os.utime(b, (1_000_001_000, 1_000_001_000))

    pending = find_pending(tmp_path)
    assert pending == [a, b]


def test_find_pending_ignores_attachments_and_dotfolders(tmp_path: Path) -> None:
    _write(tmp_path / "AI" / "attachments" / "x.md", "---\nreview: pending\n---\n\nx\n")
    _write(tmp_path / ".tg-obsidian-bot" / "y.md", "---\nreview: pending\n---\n\ny\n")
    _write(tmp_path / "AI" / "z.md", "---\nreview: pending\n---\n\nz\n")
    pending = find_pending(tmp_path)
    assert [p.name for p in pending] == ["z.md"]


def test_find_pending_skips_unavailable_icloud_file(tmp_path: Path, monkeypatch) -> None:
    available = tmp_path / "AI" / "available.md"
    unavailable = tmp_path / "AI" / "unavailable.md"
    _write(available, "---\nreview: pending\n---\n\navailable\n")
    _write(unavailable, "---\nreview: pending\n---\n\nunavailable\n")
    monkeypatch.setattr(
        inbox_module, "is_available", lambda path: path != unavailable
    )

    assert find_pending(tmp_path) == [available]


def test_clear_pending_removes_field(tmp_path: Path) -> None:
    note = tmp_path / "AI" / "n.md"
    _write(
        note,
        "---\ncreated: 2026-05-11T10:00:00\nreview: pending\nsource: telegram\n---\n\nbody\n",
    )
    changed = clear_pending(note)
    assert changed
    text = note.read_text(encoding="utf-8")
    assert "review:" not in text
    assert "created: 2026-05-11T10:00:00" in text
    assert "source: telegram" in text
    assert "body" in text
    # Idempotent
    assert clear_pending(note) is False


def test_move_note_to_other_folder(tmp_path: Path) -> None:
    src = tmp_path / "00 - Inbox" / "note.md"
    _write(src, "x\n")
    dest_dir = tmp_path / "AI"
    moved = move_note(src, dest_dir)
    assert moved == dest_dir / "note.md"
    assert moved.exists()
    assert not src.exists()


def test_move_note_avoids_filename_collision(tmp_path: Path) -> None:
    src = tmp_path / "00 - Inbox" / "note.md"
    existing = tmp_path / "AI" / "note.md"
    _write(src, "new\n")
    _write(existing, "old\n")
    moved = move_note(src, tmp_path / "AI")
    assert moved.name == "note (2).md"
    assert existing.read_text() == "old\n"
    assert moved.read_text() == "new\n"


def test_preview_returns_first_body_lines(tmp_path: Path) -> None:
    note = tmp_path / "AI" / "n.md"
    _write(
        note,
        "---\nreview: pending\n---\n\nline 1\nline 2\n\nline 3\nline 4\nline 5\nline 6\nline 7\n",
    )
    out = preview(note, max_lines=4)
    assert "line 1" in out
    assert "line 4" in out
    assert "line 7" not in out
