from __future__ import annotations

import hashlib
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np

from .config import EMBEDDING_DIM, EMBEDDING_MODEL
from .vault import (
    FRONTMATTER_RE,
    SNIPPET_WINDOW,
    SearchHit,
    _extract_tags,
    is_available,
    _strip_frontmatter,
    search_vault,
)

log = logging.getLogger(__name__)

DB_RELATIVE_PATH = Path(".tg-obsidian-bot") / "embeddings.db"
EMBED_BATCH_SIZE = 64
SURFACE_MAX_CHARS = 2000
IGNORE_DIRS = ("attachments",)


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    """Real embedder backed by OpenAI text-embedding-3-small."""

    def __init__(self, api_key: str, model: str = EMBEDDING_MODEL, dim: int = EMBEDDING_DIM):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [d.embedding for d in resp.data]


def _surface_text(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return path.stem
    fm = FRONTMATTER_RE.match(text)
    body = text[fm.end():] if fm else text
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()][:2]
    tags = _extract_tags(text)
    parts: list[str] = [path.stem]
    if tags:
        parts.append("Tags: " + ", ".join(tags))
    parts.extend(paragraphs)
    joined = "\n".join(parts).strip()
    return joined[:SURFACE_MAX_CHARS] if len(joined) > SURFACE_MAX_CHARS else joined


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _vector_to_blob(vec: list[float] | np.ndarray) -> bytes:
    arr = np.asarray(vec, dtype=np.float32)
    return arr.tobytes()


def _blob_to_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


@dataclass
class _CachedRow:
    mtime: float
    text_hash: str


@dataclass
class NearestHit:
    path: Path
    score: float


class SemanticIndex:
    """Per-note vector store backed by SQLite, with cosine nearest-neighbour search.

    Embeds `title + tags + first 2 paragraphs` per note. Caches by (mtime, content hash)
    so unchanged notes are not re-embedded across runs.
    """

    def __init__(
        self,
        base_dir: Path,
        embedder: Embedder | None,
        *,
        ignore_dirs: tuple[str, ...] = IGNORE_DIRS,
        db_path: Path | None = None,
    ):
        self.base_dir = Path(base_dir)
        self.embedder = embedder
        self.ignore_dirs = ignore_dirs
        self.db_path = db_path or (self.base_dir / DB_RELATIVE_PATH)
        self._paths: list[Path] = []
        self._matrix: np.ndarray | None = None  # normalized (N, dim)
        self._dim: int | None = embedder.dim if embedder is not None else None
        self._ensure_db()

    @property
    def enabled(self) -> bool:
        return self.embedder is not None

    def _ensure_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    path TEXT PRIMARY KEY,
                    mtime REAL NOT NULL,
                    text_hash TEXT NOT NULL,
                    dim INTEGER NOT NULL,
                    vector BLOB NOT NULL
                )
                """
            )

    def _load_cache(self) -> dict[str, _CachedRow]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT path, mtime, text_hash FROM embeddings"
            ).fetchall()
        return {p: _CachedRow(mtime=m, text_hash=h) for p, m, h in rows}

    def _iter_vault_notes(self) -> Iterable[Path]:
        for path in self.base_dir.rglob("*.md"):
            rel_parts = path.relative_to(self.base_dir).parts
            if any(part in self.ignore_dirs for part in rel_parts):
                continue
            if rel_parts and rel_parts[0].startswith("."):
                # Skip hidden directories like our own .tg-obsidian-bot/
                continue
            if not is_available(path):
                log.warning("Skipping unavailable iCloud file: %s", path)
                continue
            yield path

    def refresh(self) -> None:
        """Sync the SQLite cache to the current vault state. Idempotent."""
        if not self.enabled:
            self._paths = []
            self._matrix = None
            return

        cached = self._load_cache()
        current_rel: list[str] = []
        to_embed: list[tuple[str, float, str, str]] = []  # (rel, mtime, hash, surface)

        for path in self._iter_vault_notes():
            rel = path.relative_to(self.base_dir).as_posix()
            current_rel.append(rel)
            mtime = path.stat().st_mtime
            surface = _surface_text(path)
            text_hash = _hash(surface)
            row = cached.get(rel)
            if row is None or row.text_hash != text_hash:
                to_embed.append((rel, mtime, text_hash, surface))

        if to_embed:
            log.info("SemanticIndex: embedding %d new/changed notes", len(to_embed))
            with sqlite3.connect(self.db_path) as conn:
                for i in range(0, len(to_embed), EMBED_BATCH_SIZE):
                    batch = to_embed[i : i + EMBED_BATCH_SIZE]
                    vectors = self.embedder.embed([row[3] for row in batch])
                    if len(vectors) != len(batch):
                        raise RuntimeError(
                            f"Embedder returned {len(vectors)} vectors for "
                            f"{len(batch)} inputs"
                        )
                    for (rel, mtime, text_hash, _), vec in zip(batch, vectors):
                        conn.execute(
                            "INSERT OR REPLACE INTO embeddings "
                            "(path, mtime, text_hash, dim, vector) "
                            "VALUES (?, ?, ?, ?, ?)",
                            (rel, mtime, text_hash, len(vec), _vector_to_blob(vec)),
                        )
                conn.commit()

        # Prune deleted notes.
        current_set = set(current_rel)
        stale = [rel for rel in cached if rel not in current_set]
        if stale:
            with sqlite3.connect(self.db_path) as conn:
                conn.executemany(
                    "DELETE FROM embeddings WHERE path = ?",
                    [(r,) for r in stale],
                )
                conn.commit()

        self._reload_matrix()

    def _reload_matrix(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT path, vector FROM embeddings ORDER BY path"
            ).fetchall()
        if not rows:
            self._paths = []
            self._matrix = None
            return
        self._paths = [self.base_dir / r[0] for r in rows]
        vectors = np.vstack([_blob_to_vector(r[1]) for r in rows])
        self._dim = vectors.shape[1]
        self._matrix = _normalize(vectors.astype(np.float32))

    def nearest(self, query: str, k: int = 5) -> list[NearestHit]:
        if not self.enabled or self._matrix is None or not self._paths:
            return []
        query_vec = self.embedder.embed([query])
        if not query_vec:
            return []
        q = np.asarray(query_vec[0], dtype=np.float32)
        if q.shape[0] != self._matrix.shape[1]:
            log.warning(
                "SemanticIndex: query dim %d != stored dim %d; rebuild cache",
                q.shape[0], self._matrix.shape[1],
            )
            return []
        q = q / (np.linalg.norm(q) or 1.0)
        scores = self._matrix @ q
        if k >= len(scores):
            order = np.argsort(-scores)
        else:
            top = np.argpartition(-scores, k)[:k]
            order = top[np.argsort(-scores[top])]
        return [NearestHit(path=self._paths[i], score=float(scores[i])) for i in order]

    def nearest_excluding(
        self, query: str, exclude: Path, k: int = 5
    ) -> list[NearestHit]:
        hits = self.nearest(query, k=k + 1)
        return [h for h in hits if h.path != exclude][:k]

    def nearest_for_path(self, path: Path, k: int = 5) -> list[NearestHit]:
        """Use the cached vector for `path` (no re-embedding) to find neighbours."""
        if not self.enabled or self._matrix is None or not self._paths:
            return []
        try:
            idx = self._paths.index(path)
        except ValueError:
            return []
        q = self._matrix[idx]
        scores = self._matrix @ q
        scores[idx] = -np.inf  # exclude self
        take = min(k, len(scores) - 1)
        if take <= 0:
            return []
        top = np.argpartition(-scores, take)[:take]
        order = top[np.argsort(-scores[top])]
        return [NearestHit(path=self._paths[i], score=float(scores[i])) for i in order]


RRF_K = 60


def _category_for(path: Path, base_dir: Path) -> str:
    if path.parent == base_dir:
        return "/"
    return path.parent.relative_to(base_dir).as_posix()


def _semantic_snippet(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    body = _strip_frontmatter(text).strip()
    if not body:
        return ""
    return " ".join(body.split())[: SNIPPET_WINDOW * 2]


def hybrid_search(
    base_dir: Path,
    query: str,
    k: int,
    semantic_index: SemanticIndex | None = None,
) -> list[SearchHit]:
    """BM25-only when semantic_index is None or disabled; otherwise RRF fusion of both."""
    bm25 = search_vault(base_dir, query, k=max(k, 10))
    if semantic_index is None or not semantic_index.enabled:
        return bm25[:k]

    semantic = semantic_index.nearest(query, k=max(k, 10))
    if not semantic and not bm25:
        return []

    # RRF: aggregate reciprocal ranks from both lists.
    scores: dict[Path, float] = {}
    bm25_lookup: dict[Path, SearchHit] = {h.path: h for h in bm25}
    for rank, hit in enumerate(bm25):
        scores[hit.path] = scores.get(hit.path, 0.0) + 1.0 / (RRF_K + rank)
    for rank, hit in enumerate(semantic):
        scores[hit.path] = scores.get(hit.path, 0.0) + 1.0 / (RRF_K + rank)

    ordered = sorted(scores.items(), key=lambda x: -x[1])[:k]
    hits: list[SearchHit] = []
    for path, fused in ordered:
        existing = bm25_lookup.get(path)
        if existing is not None:
            hits.append(
                SearchHit(
                    path=existing.path,
                    title=existing.title,
                    snippet=existing.snippet,
                    score=existing.score,
                    category=existing.category,
                )
            )
        else:
            hits.append(
                SearchHit(
                    path=path,
                    title=path.stem,
                    snippet=_semantic_snippet(path),
                    score=0,
                    category=_category_for(path, base_dir),
                )
            )
    return hits
