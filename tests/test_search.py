from __future__ import annotations

from pathlib import Path

from engram import vault as vault_module
from engram.vault import load_note_body, search_vault


def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_search_matches_title_tag_and_body(tmp_path: Path) -> None:
    _write(tmp_path / "AI" / "Transformers Explained.md", "# T\nbody about attention\n#ml\n")
    _write(tmp_path / "Crypto" / "Solana Outage.md", "---\ntags: [crypto, solana]\n---\n# S\noutage details\n")
    _write(tmp_path / "Other" / "Random Thoughts.md", "completely unrelated content\n")
    _write(tmp_path / "attachments" / "ignored.md", "transformers everywhere\n")

    hits = search_vault(tmp_path, "transformers")
    titles = [h.title for h in hits]
    assert "Transformers Explained" in titles
    assert "ignored" not in titles

    hits = search_vault(tmp_path, "solana")
    assert any(h.title == "Solana Outage" for h in hits)
    # The match on tag + title + body should rank well.
    assert hits[0].title == "Solana Outage"

    assert search_vault(tmp_path, "nonexistent_term_xyz") == []


def test_search_snippet_contains_query(tmp_path: Path) -> None:
    body = "lorem ipsum " * 30 + "needle here\n" + "dolor sit " * 30
    _write(tmp_path / "Note.md", f"---\ntags: [x]\n---\n{body}")
    hits = search_vault(tmp_path, "needle")
    assert hits and "needle" in hits[0].snippet.lower()


def test_search_categorizes_by_subfolder(tmp_path: Path) -> None:
    _write(tmp_path / "AI" / "n.md", "alpha beta gamma\n")
    _write(tmp_path / "n2.md", "alpha\n")
    hits = search_vault(tmp_path, "alpha")
    cats = {h.title: h.category for h in hits}
    assert cats["n"] == "AI"
    assert cats["n2"] == "/"


def test_load_note_body_strips_frontmatter(tmp_path: Path) -> None:
    p = tmp_path / "x.md"
    p.write_text("---\ntags: [a]\n---\n\nthe real body\n", encoding="utf-8")
    assert load_note_body(p).strip() == "the real body"


def test_load_note_body_truncates(tmp_path: Path) -> None:
    p = tmp_path / "x.md"
    p.write_text("a" * 10000, encoding="utf-8")
    body = load_note_body(p, max_chars=500)
    assert "[truncated]" in body
    assert len(body) <= 600


def test_load_note_body_skips_unavailable_icloud_file(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "unavailable.md"
    _write(path, "this must not be read")
    monkeypatch.setattr(vault_module, "is_available", lambda _: False)

    assert load_note_body(path) == ""
