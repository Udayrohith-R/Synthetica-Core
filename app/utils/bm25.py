"""Pure-Python BM25 (Okapi) for Module 1 lexical retrieval."""

from __future__ import annotations

import math
from collections import Counter

from app.utils.entropy import tokenize


class BM25Index:
    """
    In-memory Okapi BM25 index.

    Production deployments can swap this for a Rust SIMD lexical backend;
    the public surface (add / search) stays identical.
    """

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: list[list[str]] = []
        self._doc_ids: list[str] = []
        self._df: Counter[str] = Counter()
        self._avgdl: float = 0.0

    def __len__(self) -> int:
        return len(self._docs)

    def add(self, doc_id: str, text: str) -> None:
        tokens = tokenize(text)
        self._doc_ids.append(doc_id)
        self._docs.append(tokens)
        self._df.update(set(tokens))
        total_len = sum(len(d) for d in self._docs)
        self._avgdl = total_len / max(len(self._docs), 1)

    def add_many(self, items: list[tuple[str, str]]) -> None:
        for doc_id, text in items:
            self.add(doc_id, text)

    def search(self, query: str, *, top_k: int = 10) -> list[tuple[str, float]]:
        if not self._docs:
            return []
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        n = len(self._docs)
        scores: list[tuple[str, float]] = []
        for doc_id, tokens in zip(self._doc_ids, self._docs, strict=True):
            tf = Counter(tokens)
            dl = len(tokens) or 1
            score = 0.0
            for term in q_tokens:
                if term not in tf:
                    continue
                df = self._df[term]
                idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
                freq = tf[term]
                denom = freq + self.k1 * (1.0 - self.b + self.b * dl / max(self._avgdl, 1e-9))
                score += idf * (freq * (self.k1 + 1.0)) / denom
            if score > 0:
                scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
