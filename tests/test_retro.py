from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from engram import retro as retro_module
from engram.embeddings import NearestHit
from engram.retro import merge_duplicate_notes


class FakeIndex:
    """nearest() returns the configured hit for the given query note path."""

    def __init__(self, mapping: dict[str, list[tuple[Path, float]]]):
        # mapping: query note stem -> ranked (path, score)
        self._mapping = mapping
        self.enabled = True

    def nearest(self, query: str, k: int = 5):
        first_line = query.splitlines()[0] if query else ""
        for stem, ranked in self._mapping.items():
            if stem in query:
                return [NearestHit(path=p, score=s) for p, s in ranked][:k]
        return []


def _client(merged_body: str) -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=merged_body)]
    )
    return client


def test_disabled_index_is_noop(tmp_path: Path) -> None:
    class Off:
        enabled = False

    assert merge_duplicate_notes(tmp_path, Off(), MagicMock(), []) == 0


def test_merges_duplicate_keeps_older_as_canonical(tmp_path: Path) -> None:
    import os

    canonical = tmp_path / "AI" / "Bitter Lesson.md"
    dup = tmp_path / "AI" / "scaling laws.md"
    canonical.parent.mkdir(parents=True)
    canonical.write_text("---\nc: x\n---\n# Bitter Lesson\ncanonical body\n", encoding="utf-8")
    dup.write_text("---\nc: y\n---\n# scaling laws\nduplicate body\n", encoding="utf-8")
    # Make canonical clearly older so it survives the merge.
    os.utime(canonical, (1000, 1000))
    os.utime(dup, (2000, 2000))

    idx = FakeIndex({"scaling laws": [(canonical, 0.95)], "Bitter Lesson": [(dup, 0.95)]})
    client = _client("---\nc: x\n---\n# Bitter Lesson\nmerged canonical + duplicate\n")

    merges = merge_duplicate_notes(tmp_path, idx, client, [canonical, dup])

    assert merges == 1
    assert canonical.exists() and not dup.exists()
    assert "merged canonical + duplicate" in canonical.read_text(encoding="utf-8")


def test_unsafe_merge_keeps_both(tmp_path: Path) -> None:
    a = tmp_path / "AI" / "a.md"
    b = tmp_path / "AI" / "b.md"
    a.parent.mkdir(parents=True)
    a.write_text("---\nc: x\n---\n# a\nbody a\n", encoding="utf-8")
    b.write_text("---\nc: y\n---\n# b\nbody b\n", encoding="utf-8")

    idx = FakeIndex({"a": [(b, 0.95)], "b": [(a, 0.95)]})
    # Returned merge dropped frontmatter → unsafe → skip, nothing deleted.
    client = _client("# a\nno frontmatter\n")

    merges = merge_duplicate_notes(tmp_path, idx, client, [a, b])

    assert merges == 0
    assert a.exists() and b.exists()


def test_no_duplicate_is_noop(tmp_path: Path) -> None:
    a = tmp_path / "AI" / "a.md"
    a.parent.mkdir(parents=True)
    a.write_text("---\nc: x\n---\n# a\nbody a\n", encoding="utf-8")
    idx = FakeIndex({"a": [(a, 0.3)]})  # only itself, low score

    assert merge_duplicate_notes(tmp_path, idx, MagicMock(), [a]) == 0
    assert a.exists()


def test_merge_skips_unavailable_icloud_file(tmp_path: Path, monkeypatch) -> None:
    available = tmp_path / "AI" / "available.md"
    unavailable = tmp_path / "AI" / "unavailable.md"
    available.parent.mkdir()
    available.write_text("---\nc: x\n---\n# Available\nbody\n", encoding="utf-8")
    unavailable.write_text(
        "---\nc: y\n---\n# Unavailable\nbody\n", encoding="utf-8"
    )
    index = FakeIndex({"available": [(unavailable, 0.95)]})
    monkeypatch.setattr(
        retro_module, "is_available", lambda path: path != unavailable
    )

    assert merge_duplicate_notes(
        tmp_path, index, MagicMock(), [available, unavailable]
    ) == 0
    assert available.exists()
    assert unavailable.exists()
