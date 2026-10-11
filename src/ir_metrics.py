from __future__ import annotations
import numpy as np


def per_query(ranked: dict[str, list[str]], rel: dict[str, set[str]], k: int = 10) -> dict[str, np.ndarray]:
    """Binary-relevance nDCG@k, Recall@k, Recall@100, MRR@k for every query."""
    out = {m: [] for m in ["ndcg@10", "recall@10", "recall@100", "mrr@10"]}
    for q, rs in rel.items():
        r = ranked[q]
        gains = np.array([1.0 if d in rs else 0.0 for d in r[:k]])
        disc = 1 / np.log2(np.arange(2, len(gains) + 2))
        ideal = disc[: min(len(rs), k)].sum()
        out["ndcg@10"].append((gains * disc).sum() / ideal)
        out["recall@10"].append(gains.sum() / len(rs))
        out["recall@100"].append(sum(d in rs for d in r[:100]) / len(rs))
        rr = [1 / (i + 1) for i, d in enumerate(r[:k]) if d in rs]
        out["mrr@10"].append(rr[0] if rr else 0.0)
    return {m: np.array(v) for m, v in out.items()}


def bootstrap_ci(x: np.ndarray, n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 5000, seed: int = 0):
    """Mean difference a-b with 95% CI and a two-sided bootstrap p-value (paired over queries)."""
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    p = 2 * min((m <= 0).mean(), (m >= 0).mean())
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float(min(p, 1.0))
