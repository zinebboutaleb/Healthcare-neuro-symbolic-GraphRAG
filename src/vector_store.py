"""Vanilla vector RAG: documents -> chunks -> embeddings -> FAISS -> top-k."""
from __future__ import annotations

import re
from pathlib import Path

import faiss
import numpy as np

from src.config import DOC_DIR
from src.models import DocHit


def load_documents(doc_dir: Path | str = DOC_DIR) -> list[dict]:
    """Parse `key: value` header lines, a `---` separator, then the body."""
    docs = []
    for path in sorted(Path(doc_dir).glob("*.txt")):
        head, _, body = path.read_text(encoding="utf-8").partition("\n---\n")
        meta = dict(line.split(": ", 1) for line in head.strip().splitlines())
        docs.append({**meta, "text": body.strip()})
    return docs


def chunk_text(text: str, max_chars: int = 320) -> list[str]:
    """Sentence-aware chunking: pack whole sentences until max_chars."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, cur = [], ""
    for s in sentences:
        if cur and len(cur) + len(s) + 1 > max_chars:
            chunks.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(cur)
    return chunks


class VectorStore:
    def __init__(self, embedder, docs: list[dict] | None = None, max_chars: int = 320):
        self.embedder = embedder
        self.chunks: list[dict] = []
        for d in docs if docs is not None else load_documents():
            for j, c in enumerate(chunk_text(d["text"], max_chars)):
                self.chunks.append({"chunk_id": f"{d['doc_id']}#{j}", "doc_id": d["doc_id"], "title": d["title"],
                                    "text": c, "metadata": {"doc_type": d.get("doc_type", "")}})
        # the title is prepended to the embedded text so later chunks keep their subject
        texts = [f"{c['title']}. {c['text']}" for c in self.chunks]
        if hasattr(self.embedder, "fit"):  # only the lexical fallback needs corpus statistics
            self.embedder.fit(texts)
        self.embeddings = self.embedder.embed(texts)
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])  # cosine similarity (vectors are L2-normalised)
        self.index.add(self.embeddings)

    def search(self, query: str, k: int = 4) -> list[DocHit]:
        q = self.embedder.embed([query])
        scores, ids = self.index.search(np.asarray(q, dtype="float32"), k)
        return [DocHit(rank=r + 1, score=float(s), **{x: self.chunks[i][x] for x in ("chunk_id", "doc_id", "title", "text")})
                for r, (s, i) in enumerate(zip(scores[0], ids[0])) if i >= 0]
