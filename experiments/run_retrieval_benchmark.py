"""Retrieval ablation on FiQA: BM25 vs dense vs hybrid (RRF) vs hybrid+cross-encoder vs fine-tuned dense.
Usage: python experiments/run_retrieval_benchmark.py [--finetune-pairs 3000]"""
from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sentence_transformers import SentenceTransformer, CrossEncoder
from src import fiqa, retrievers as R, ir_metrics as M

torch.set_num_threads(2)
OUT = Path("results"); OUT.mkdir(exist_ok=True); Path("data").mkdir(exist_ok=True)
BASE = "BAAI/bge-small-en-v1.5"; CE = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def main(a):
    t0 = time.time()
    D = fiqa.load(n_distractors=a.distractors)
    docs, qs, rel = D["eval_docs"], D["test_queries"], D["test_rel"]
    ids = list(docs); texts = [docs[i] for i in ids]; qids = list(rel)
    print(f"eval corpus {len(ids)} docs, {len(qids)} test queries, {len(D['train_pairs'])} train pairs", flush=True)

    results, rankings = {}, {}

    def cached(name, fn):
        """Stage cache: rankings are stored on disk so a crash never costs a finished stage."""
        import re as _re
        p = Path("data") / ("rank_" + _re.sub(r"\W+", "_", name) + ".json")
        if p.exists():
            return json.load(open(p))
        r = fn(); json.dump(r, open(p, "w")); return r

    def record(name, ranked):
        results[name] = M.per_query(ranked, rel); rankings[name] = ranked
        s = {k: round(float(v.mean()), 4) for k, v in results[name].items()}
        print(f"[{time.time()-t0:5.0f}s] {name}: {s}", flush=True)

    bm = R.BM25(ids, texts)
    bm_r = {q: [d for d, _ in bm.search(qs[q], 100)] for q in qids}
    record("BM25", bm_r)

    base = SentenceTransformer(BASE, device="cpu"); base.max_seq_length = 256
    dense = R.Dense(base, ids, texts, cache="data/emb_base.npy")
    out = dense.search_many([qs[q] for q in qids], 100)
    dn_r = {q: [d for d, _ in o] for q, o in zip(qids, out)}
    record("Dense bge-small", dn_r)

    hy_r = {q: R.rrf([bm_r[q], dn_r[q]])[:100] for q in qids}
    record("Hybrid BM25+dense (RRF)", hy_r)

    ce = CrossEncoder(CE, device="cpu", max_length=256)
    rr_r = cached("rerank_top30", lambda: {q: R.rerank(ce, qs[q], hy_r[q], docs, top=30) for q in qids})
    record("Hybrid + cross-encoder rerank (top-30)", rr_r)

    # ---- domain adaptation: fine-tune the bi-encoder with in-batch negatives on FiQA train pairs ----
    from sentence_transformers import InputExample, losses
    from torch.utils.data import DataLoader
    ft = SentenceTransformer(BASE, device="cpu"); ft.max_seq_length = 128
    ex = [InputExample(texts=[R.QUERY_PREFIX + q, d]) for q, d in D["train_pairs"][: a.finetune_pairs]]
    dl = DataLoader(ex, shuffle=True, batch_size=32, drop_last=True)
    loss = losses.MultipleNegativesRankingLoss(ft)
    print(f"[{time.time()-t0:5.0f}s] fine-tuning on {len(ex)} pairs, {len(dl)} steps", flush=True)
    import os
    if not os.path.exists("data/emb_ft.npy"):
        ft.fit(train_objectives=[(dl, loss)], epochs=1, warmup_steps=10, optimizer_params={"lr": 2e-5}, show_progress_bar=False)
    ft.max_seq_length = 256
    ft.save(str(OUT / "finetuned_bge_small")) if a.save_model else None
    dense_ft = R.Dense(ft, ids, texts, cache="data/emb_ft.npy")
    out = dense_ft.search_many([qs[q] for q in qids], 100)
    ft_r = {q: [d for d, _ in o] for q, o in zip(qids, out)}
    record("Dense bge-small fine-tuned", ft_r)
    record("Hybrid BM25+fine-tuned dense (RRF)", {q: R.rrf([bm_r[q], ft_r[q]])[:100] for q in qids})

    # ---- report ----
    rows = []
    for name, m in results.items():
        row = {"method": name}
        for k, v in m.items():
            mean, lo, hi = M.bootstrap_ci(v); row[k] = round(mean, 4); row[k + "_ci"] = [round(lo, 4), round(hi, 4)]
        rows.append(row)
    ref = results["BM25"]["ndcg@10"]
    sig = {n: dict(zip(["delta", "ci_lo", "ci_hi", "p"], map(lambda x: round(x, 4), M.paired_bootstrap(m["ndcg@10"], ref))))
           for n, m in results.items() if n != "BM25"}
    json.dump({"setup": {"eval_docs": len(ids), "test_queries": len(qids), "finetune_pairs": len(ex), "base_model": BASE,
                         "cross_encoder": CE, "note": "corpus subsampled: all test positives + random distractors"},
               "metrics": rows, "paired_ndcg10_vs_BM25": sig}, open(OUT / "retrieval_benchmark.json", "w"), indent=2)
    print(json.dumps(sig, indent=1)); print("total seconds", round(time.time() - t0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--distractors", type=int, default=8500)
    ap.add_argument("--finetune-pairs", type=int, default=3000)
    ap.add_argument("--save-model", action="store_true")
    main(ap.parse_args())
