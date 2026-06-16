"""
Retrieval-augmented generation over the racing data.

Pipeline:

1. ``build_race_documents`` turns each race + its runners into a compact text
   "card" plus metadata. These are the retrievable chunks.
2. An **embedder** maps text to vectors. Two backends:
     * ``TfidfEmbedder``  -- local, no API key, good enough for keyword-ish recall.
       This is the default so the repo runs anywhere.
     * ``DenseEmbedder``  -- a small interface to drop in a real embeddings provider
       (Voyage / OpenAI / Cohere / a local sentence-transformer). Implement
       ``embed(texts) -> np.ndarray`` and pass an instance to ``RetrievalIndex``.
3. ``RetrievalIndex`` stores vectors and answers nearest-neighbour queries
   (cosine similarity).
4. ``rag_answer`` retrieves the top-k cards for a question and asks the LLM to
   answer **using only the retrieved context** -- the standard anti-hallucination
   contract. Without an API key it returns the retrieved context directly.

Swapping TF-IDF for dense embeddings is a one-line change at call sites; the
index and RAG logic are agnostic to the backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..data import schema
from .explain import LLMClient, _SYSTEM


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #
@dataclass
class Doc:
    id: str
    text: str
    meta: dict


def build_race_documents(merged: pd.DataFrame, max_races: int | None = None) -> list[Doc]:
    """Create one retrievable text card per race from the merged frame."""
    docs: list[Doc] = []
    races = merged[schema.RACE_KEY].unique()
    if max_races:
        races = races[:max_races]
    for rid in races:
        g = merged[merged[schema.RACE_KEY] == rid]
        head = g.iloc[0]
        winners = g.sort_values("result").head(3)
        win_str = ", ".join(
            f"horse {int(r.horse_id)} (draw {int(r.draw)}, "
            f"{r.win_odds if pd.notna(r.win_odds) else '?'})"
            for r in winners.itertuples()
        )
        text = (
            f"Race {int(rid)} at {head.get('venue','?')} on "
            f"{pd.to_datetime(head.get('date')).date() if pd.notna(head.get('date')) else '?'}. "
            f"Distance {head.get('distance','?')}m, going {head.get('going','?')}, "
            f"class {head.get('race_class','?')}, {len(g)} runners. "
            f"Top finishers: {win_str}."
        )
        docs.append(Doc(
            id=f"race-{int(rid)}",
            text=text,
            meta={
                "race_id": int(rid),
                "venue": head.get("venue"),
                "distance": head.get("distance"),
                "going": head.get("going"),
                "date": str(pd.to_datetime(head.get("date")).date())
                if pd.notna(head.get("date")) else None,
            },
        ))
    return docs


# --------------------------------------------------------------------------- #
# Embedders
# --------------------------------------------------------------------------- #
class Embedder(Protocol):
    def fit(self, texts: list[str]) -> "Embedder": ...
    def embed(self, texts: list[str]) -> np.ndarray: ...


class TfidfEmbedder:
    """Local TF-IDF embedder (default). No external dependencies or API keys."""

    def __init__(self, max_features: int = 4096):
        self.vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        self._fitted = False

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        self.vec.fit(texts)
        self._fitted = True
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            self.fit(texts)
        return self.vec.transform(texts).toarray()


class DenseEmbedder:
    """Adapter for a real dense-embedding provider.

    Pass a callable ``embed_fn(list[str]) -> np.ndarray`` (e.g. wrapping Voyage AI,
    OpenAI, Cohere, or a local sentence-transformer). Example::

        import voyageai
        vo = voyageai.Client()
        embedder = DenseEmbedder(lambda xs: np.array(
            vo.embed(xs, model="voyage-3").embeddings))

    Dense vectors typically give far better semantic recall than TF-IDF.
    """

    def __init__(self, embed_fn):
        self.embed_fn = embed_fn

    def fit(self, texts: list[str]) -> "DenseEmbedder":
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.embed_fn(texts), dtype=float)


# --------------------------------------------------------------------------- #
# Index + RAG
# --------------------------------------------------------------------------- #
class RetrievalIndex:
    """In-memory cosine-similarity index over document embeddings."""

    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or TfidfEmbedder()
        self.docs: list[Doc] = []
        self._mat: np.ndarray | None = None

    def build(self, docs: list[Doc]) -> "RetrievalIndex":
        self.docs = docs
        texts = [d.text for d in docs]
        self.embedder.fit(texts)
        self._mat = self.embedder.embed(texts)
        return self

    def query(self, text: str, k: int = 5) -> list[tuple[Doc, float]]:
        if self._mat is None:
            raise RuntimeError("Index not built. Call build(docs) first.")
        q = self.embedder.embed([text])
        sims = cosine_similarity(q, self._mat)[0]
        top = np.argsort(-sims)[:k]
        return [(self.docs[i], float(sims[i])) for i in top]


def rag_answer(
    question: str,
    index: RetrievalIndex,
    client: LLMClient | None = None,
    k: int = 5,
) -> str:
    """Answer a question using retrieved race cards as the only context."""
    hits = index.query(question, k=k)
    context = "\n".join(f"[{d.id}] {d.text}" for d, _ in hits)

    if client and client.available:
        prompt = (
            f"Question: {question}\n\n"
            f"Context (retrieved race cards — use ONLY these):\n{context}\n\n"
            "Answer the question grounded strictly in the context. If the context "
            "doesn't contain the answer, say so."
        )
        out = client.generate(_SYSTEM, prompt, max_tokens=600)
        if out:
            return out

    # ---- offline fallback: return the retrieved evidence ----
    lines = [f"(No LLM key set — returning the {len(hits)} most relevant race cards.)"]
    lines += [f"- {d.text}  [score {s:.2f}]" for d, s in hits]
    return "\n".join(lines)
