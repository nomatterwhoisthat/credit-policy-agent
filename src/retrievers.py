"""BM25, dense bi-encoder, reciprocal-rank-fusion hybrid and cross-encoder reranking."""
from __future__ import annotations
import re
import numpy as np
from rank_bm25 import BM25Okapi

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "   # BGE retrieval instruction
_tok = re.compile(r"[a-z0-9]+")


def tokenize(t: str) -> list[str]:
    return _tok.findall(t.lower())


class BM25:
    def __init__(self, ids, texts):
        self.ids = np.array(ids); self.bm = BM25Okapi([tokenize(t) for t in texts])

    def search(self, q: str, k: int = 100):
        s = self.bm.get_scores(tokenize(q)); top = np.argsort(-s)[:k]
        return [(self.ids[i], float(s[i])) for i in top]


class Dense:
    def __init__(self, model, ids, texts, prefix: str = QUERY_PREFIX, batch: int = 64, cache: str | None = None):
        import os
        self.m, self.ids, self.prefix = model, np.array(ids), prefix
        if cache and os.path.exists(cache):
            self.E = np.load(cache)
        else:
            self.E = model.encode(texts, batch_size=batch, normalize_embeddings=True, show_progress_bar=False)
            if cache:
                np.save(cache, self.E)

    def encode_queries(self, qs):
        return self.m.encode([self.prefix + q for q in qs], batch_size=64, normalize_embeddings=True, show_progress_bar=False)

    def search_many(self, qs, k: int = 100):
        S = self.encode_queries(qs) @ self.E.T
        top = np.argsort(-S, axis=1)[:, :k]
        return [[(self.ids[i], float(S[r, i])) for i in top[r]] for r in range(len(qs))]


def rrf(rankings: list[list[str]], k: int = 60, depth: int = 100) -> list[str]:
    """Reciprocal rank fusion (Cormack et al., 2009)."""
    sc: dict[str, float] = {}
    for r in rankings:
        for rank, d in enumerate(r[:depth]):
            sc[d] = sc.get(d, 0.0) + 1.0 / (k + rank + 1)
    return [d for d, _ in sorted(sc.items(), key=lambda x: -x[1])]


def rerank(ce, query: str, cand_ids: list[str], docs: dict[str, str], top: int = 30, max_len: int = 256):
    head, tail = cand_ids[:top], cand_ids[top:]
    scores = ce.predict([(query, docs[d]) for d in head], batch_size=32, show_progress_bar=False)
    return [d for _, d in sorted(zip(-np.array(scores), head))] + tail
