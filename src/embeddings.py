"""Embedding backends. Default: sentence-transformers/all-MiniLM-L6-v2.

EMBEDDING_BACKEND = auto | minilm | hashing
  auto    -> MiniLM if sentence-transformers + the model are available, else the hashing fallback (warns).
  hashing -> dependency-free bag-of-words/bigram hashing embedder. NOT a semantic model: it exists so the
             project runs offline (CI, no model download). Report which backend produced any results.
"""
from __future__ import annotations

import logging
import re
import zlib
from functools import lru_cache

import numpy as np

from src.config import env

log = logging.getLogger(__name__)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_STOP = set("a an the of and or in on to for with is are be as by at it its this that these those from into include includes listed reported".split())


class HashingEmbedder:
    """TF-IDF over hashed unigrams+bigrams. `fit()` learns IDF weights from the corpus (called by VectorStore)."""

    name = "tfidf-hashing-fallback"
    dim = 2048

    def __init__(self):
        self.idf = np.ones(self.dim, dtype="float32")

    def _tf(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            toks = [w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in _STOP]
            for f in toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]:
                out[i, zlib.crc32(f.encode()) % self.dim] += 1.0
        return np.log1p(out)

    def fit(self, texts: list[str]) -> None:
        df = (self._tf(texts) > 0).sum(axis=0)
        self.idf = (np.log((1 + len(texts)) / (1 + df)) + 1.0).astype("float32")

    def embed(self, texts: list[str]) -> np.ndarray:
        out = self._tf(texts) * self.idf
        return out / np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-9, None)


class MiniLMEmbedder:
    name = "all-MiniLM-L6-v2"

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(MODEL_NAME)
        self.dim = getattr(self.model, "get_embedding_dimension", self.model.get_sentence_embedding_dimension)()

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype="float32")


@lru_cache(maxsize=4)
def _build(backend: str):
    if backend == "hashing":
        return HashingEmbedder()
    try:
        return MiniLMEmbedder()
    except Exception as exc:  # noqa: BLE001
        if backend == "minilm":
            raise
        log.warning("MiniLM unavailable (%s) - using the hashing fallback embedder (not semantic).", exc)
        return HashingEmbedder()


def get_embedder(backend: str | None = None):
    return _build((backend or env("EMBEDDING_BACKEND", "auto")).lower())