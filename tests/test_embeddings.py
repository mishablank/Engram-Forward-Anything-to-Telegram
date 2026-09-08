from __future__ import annotations

import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from engram.embeddings import SemanticIndex, _surface_text
from engram.vault import scan_vault, search_vault


class StubEmbedder:
    """Deterministic embedder for tests: vectors derived from token bag."""

    dim = 32

    def __init__(self):
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        out: list[list[float]] = []
        for t in texts:
            vec = [0.0] * self.dim
            for token in t.lower().split():
                h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
                vec[h % self.dim] += 1.0
            out.append(vec)
        return out


def _write(p: Path, body: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


def test_iter_vault_notes_skips_dataless_files(tmp_path: Path, monkeypatch) -> None:
    available = tmp_path / "AI" / "available.md"
    dataless = tmp_path / "AI" / "dataless.md"
    _write(available, "available\n")
    _write(dataless, "unavailable\n")
    real_stat = Path.stat

    def fake_stat(path: Path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        flags = 0x40000000 if path == dataless else 0
        return SimpleNamespace(
            st_flags=flags,
            st_mode=result.st_mode,
            st_mtime=result.st_mtime,
        )

    monkeypatch.setattr(Path, "stat", fake_stat)
    idx = SemanticIndex(tmp_path, None)

    assert list(idx._iter_vault_notes()) == [available]
    assert scan_vault(tmp_path).titles == ["available"]
    assert [hit.title for hit in search_vault(tmp_path, "available")] == ["available"]


def test_vault_indexes_skip_stat_errors(tmp_path: Path, monkeypatch) -> None:
    available = tmp_path / "AI" / "available.md"
    unavailable = tmp_path / "AI" / "unavailable.md"
    _write(available, "available\n")
    _write(unavailable, "unavailable\n")
    real_stat = Path.stat

    def fake_stat(path: Path, *args, **kwargs):
        if path == unavailable:
            raise OSError("iCloud file unavailable")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fake_stat)
    embedder = StubEmbedder()
    idx = SemanticIndex(tmp_path, embedder)
    idx.refresh()

    assert list(idx._iter_vault_notes()) == [available]
    assert scan_vault(tmp_path).titles == ["available"]
    assert [hit.title for hit in search_vault(tmp_path, "available")] == ["available"]
    assert embedder.calls == [["available\navailable"]]


def test_surface_text_uses_title_tags_first_paragraphs(tmp_path: Path) -> None:
    note = tmp_path / "AI" / "Sora 2 Launch.md"
    _write(
        note,
        "---\ntags: [ai, video]\n---\n\nFirst paragraph here.\n\nSecond paragraph.\n\n"
        "Third paragraph should be skipped.\n",
    )
    surface = _surface_text(note)
    assert "Sora 2 Launch" in surface
    assert "First paragraph here." in surface
    assert "Second paragraph." in surface
    assert "Third paragraph" not in surface


def test_refresh_embeds_each_note_once_when_unchanged(tmp_path: Path) -> None:
    _write(tmp_path / "AI" / "alpha.md", "alpha gamma\n")
    _write(tmp_path / "AI" / "beta.md", "beta delta\n")
    emb = StubEmbedder()
    idx = SemanticIndex(tmp_path, emb)
    idx.refresh()
    first_call_count = sum(len(c) for c in emb.calls)
    assert first_call_count == 2

    # Second refresh with no changes should re-embed nothing.
    emb.calls.clear()
    idx.refresh()
    assert emb.calls == []


def test_refresh_reembeds_on_content_change(tmp_path: Path) -> None:
    note = tmp_path / "AI" / "alpha.md"
    _write(note, "alpha gamma\n")
    emb = StubEmbedder()
    idx = SemanticIndex(tmp_path, emb)
    idx.refresh()
    emb.calls.clear()

    _write(note, "alpha epsilon zeta\n")
    idx.refresh()
    assert len(emb.calls) == 1
    assert emb.calls[0] == ["alpha\nalpha epsilon zeta"]


def test_refresh_prunes_deleted_notes(tmp_path: Path) -> None:
    a = tmp_path / "AI" / "alpha.md"
    b = tmp_path / "AI" / "beta.md"
    _write(a, "alpha\n")
    _write(b, "beta\n")
    emb = StubEmbedder()
    idx = SemanticIndex(tmp_path, emb)
    idx.refresh()

    b.unlink()
    idx.refresh()
    hits = idx.nearest("beta", k=5)
    paths = [h.path for h in hits]
    assert b not in paths


def test_nearest_ranks_by_cosine_similarity(tmp_path: Path) -> None:
    _write(tmp_path / "AI" / "sora launch.md", "sora two launched in october\n")
    _write(tmp_path / "AI" / "crypto news.md", "bitcoin halving mining rewards\n")
    _write(tmp_path / "AI" / "gardening.md", "tomatoes basil compost watering\n")
    emb = StubEmbedder()
    idx = SemanticIndex(tmp_path, emb)
    idx.refresh()

    hits = idx.nearest("sora two october", k=2)
    assert hits[0].path.stem == "sora launch"


def test_nearest_with_empty_vault_returns_empty(tmp_path: Path) -> None:
    emb = StubEmbedder()
    idx = SemanticIndex(tmp_path, emb)
    idx.refresh()
    assert idx.nearest("anything", k=5) == []


def test_disabled_when_no_embedder(tmp_path: Path) -> None:
    _write(tmp_path / "AI" / "alpha.md", "alpha\n")
    idx = SemanticIndex(tmp_path, embedder=None)
    idx.refresh()
    assert not idx.enabled
    assert idx.nearest("alpha", k=5) == []


def test_ignores_attachments_and_dotfolders(tmp_path: Path) -> None:
    _write(tmp_path / "AI" / "alpha.md", "alpha\n")
    _write(tmp_path / "AI" / "attachments" / "note.md", "should be ignored\n")
    _write(tmp_path / ".tg-obsidian-bot" / "hidden.md", "should be ignored too\n")
    emb = StubEmbedder()
    idx = SemanticIndex(tmp_path, emb)
    idx.refresh()
    paths = [p.name for p in idx._paths]
    assert paths == ["alpha.md"]


def test_hybrid_search_falls_back_to_bm25_when_no_semantic(tmp_path: Path) -> None:
    from engram.embeddings import hybrid_search

    _write(tmp_path / "AI" / "alpha.md", "alpha gamma\n")
    _write(tmp_path / "Other" / "beta.md", "beta delta\n")
    hits = hybrid_search(tmp_path, "alpha", k=3, semantic_index=None)
    assert [h.title for h in hits] == ["alpha"]


def test_hybrid_search_merges_bm25_and_semantic(tmp_path: Path) -> None:
    from engram.embeddings import hybrid_search

    # BM25 will match this on exact term.
    _write(tmp_path / "AI" / "bm25_match.md", "rocket launch propulsion\n")
    # Semantic-only candidate: shares no exact token with the query.
    _write(tmp_path / "AI" / "semantic_only.md", "rocket launch propulsion details here\n")
    # An unrelated note.
    _write(tmp_path / "Other" / "noise.md", "tomatoes basil compost\n")

    emb = StubEmbedder()
    idx = SemanticIndex(tmp_path, emb)
    idx.refresh()

    hits = hybrid_search(tmp_path, "rocket propulsion", k=3, semantic_index=idx)
    titles = [h.title for h in hits]
    assert "bm25_match" in titles
    # And we should get a semantic hit alongside it.
    assert len(hits) >= 2


def test_nearest_excluding_skips_given_path(tmp_path: Path) -> None:
    a = tmp_path / "AI" / "a.md"
    b = tmp_path / "AI" / "b.md"
    _write(a, "alpha alpha alpha\n")
    _write(b, "alpha alpha alpha\n")
    emb = StubEmbedder()
    idx = SemanticIndex(tmp_path, emb)
    idx.refresh()

    hits = idx.nearest_excluding("alpha", exclude=a, k=2)
    assert all(h.path != a for h in hits)
    assert b in [h.path for h in hits]
