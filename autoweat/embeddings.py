"""
embeddings.py — pluggable embedding backends.

Two backends, both return a uniform (embed_fn, vocab_set) interface:

    backend = load_backend(cfg)
    vec = backend.embed("woman")
    vocab = backend.vocab

1. GensimBackend — loads a KeyedVectors file (word2vec binary, GloVe text,
   fasttext .vec). This is the Caliskan-faithful default.

2. OllamaBackend — calls a local Ollama server's /api/embeddings endpoint
   for any model tagged as an embedding model (e.g. nomic-embed-text).
   Vocabulary for an LLM embedder is unbounded in principle, so we
   maintain an on-disk allowlist built from proposed words; membership is
   decided by "have we successfully embedded this word before?"
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

import numpy as np


class EmbeddingBackend(Protocol):
    vocab: set[str]
    name: str
    def embed(self, word: str) -> np.ndarray: ...


# ---------- gensim ----------

class GensimBackend:
    """Wraps a gensim KeyedVectors model.

    Supports the three formats Caliskan et al. used and their descendants:
      - word2vec binary (.bin)
      - GloVe / word2vec text (.txt, .vec)
      - fasttext .vec
    """

    def __init__(self, path: str, binary: bool | None = None, limit: int | None = None,
                 mmap: str | None = "r"):
        from gensim.models import KeyedVectors  # lazy import

        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Embedding file not found: {p}")

        # Native gensim format (.kv) — instant load with optional mmap.
        # This is dramatically faster than re-parsing word2vec text/binary
        # on every startup. The companion .vectors.npy file must sit
        # alongside the .kv file.
        if p.suffix == ".kv":
            self.kv = KeyedVectors.load(str(p), mmap=mmap)
        else:
            # Legacy word2vec text or binary format. Slow first load.
            if binary is None:
                binary = p.suffix.lower() == ".bin"
            self.kv = KeyedVectors.load_word2vec_format(
                str(p), binary=binary, limit=limit
            )

        self.vocab = set(self.kv.key_to_index.keys())
        self.name = f"gensim:{p.name}"
        self.dim = self.kv.vector_size

    def embed(self, word: str) -> np.ndarray:
        return np.asarray(self.kv[word], dtype=np.float64)


# ---------- ollama ----------

class OllamaBackend:
    """Calls an Ollama server for embeddings.

    Maintains a JSON cache on disk so repeated embeds are free and so
    the vocab set grows as words are successfully embedded.
    """

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        cache_path: str = "data/ollama_embed_cache.json",
    ):
        import requests  # lazy import
        self.requests = requests
        self.model = model
        self.host = host.rstrip("/")
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        if self.cache_path.exists():
            with self.cache_path.open() as f:
                raw = json.load(f)
            self._cache: dict[str, list[float]] = raw
        else:
            self._cache = {}

        self.vocab = set(self._cache.keys())
        self.name = f"ollama:{model}"

    def _save(self):
        tmp = self.cache_path.with_suffix(".tmp")
        with tmp.open("w") as f:
            json.dump(self._cache, f)
        os.replace(tmp, self.cache_path)

    def embed(self, word: str) -> np.ndarray:
        if word in self._cache:
            return np.asarray(self._cache[word], dtype=np.float64)
        resp = self.requests.post(
            f"{self.host}/api/embeddings",
            json={"model": self.model, "prompt": word},
            timeout=60,
        )
        resp.raise_for_status()
        vec = resp.json()["embedding"]
        self._cache[word] = vec
        self.vocab.add(word)
        self._save()
        return np.asarray(vec, dtype=np.float64)

    def prime_vocab(self, words: list[str]) -> None:
        """Best-effort embed a batch of candidate words so they enter vocab."""
        for w in words:
            try:
                self.embed(w)
            except Exception:
                # Words that error out (e.g. whitespace, empty) just aren't vocab.
                pass


# ---------- factory ----------

def load_backend(cfg: dict) -> EmbeddingBackend:
    """Instantiate a backend from the `embedding:` block of config.yaml."""
    kind = cfg.get("backend", "gensim")
    if kind == "gensim":
        return GensimBackend(
            path=cfg["path"],
            binary=cfg.get("binary"),
            limit=cfg.get("limit"),
        )
    if kind == "ollama":
        return OllamaBackend(
            model=cfg["model"],
            host=cfg.get("host", "http://localhost:11434"),
            cache_path=cfg.get("cache_path", "data/ollama_embed_cache.json"),
        )
    raise ValueError(f"Unknown embedding backend: {kind}")
