"""FiQA-2018 (financial opinion QA) as an IR benchmark, loaded from the `mteb/fiqa` parquet mirror of BEIR/FiQA."""
from __future__ import annotations
import random
from collections import defaultdict
from datasets import load_dataset

SEED = 13


def load(n_distractors: int = 8500):
    corpus = load_dataset("mteb/fiqa", "corpus")["corpus"]
    queries = load_dataset("mteb/fiqa", "queries")["queries"]
    qrels = load_dataset("mteb/fiqa", "default")
    qtext = {r["_id"]: r["text"] for r in queries}
    dtext = {r["_id"]: ((r["title"] + " ") if r["title"] else "") + r["text"] for r in corpus}

    def rel(split):
        d = defaultdict(set)
        for r in qrels[split]:
            if r["score"] > 0:
                d[str(r["query-id"])].add(str(r["corpus-id"]))
        return d

    train, test = rel("train"), rel("test")
    dtext = {str(k): v for k, v in dtext.items()}
    qtext = {str(k): v for k, v in qtext.items()}
    test_pos = set().union(*test.values())

    # evaluation corpus = all test positives + random distractors (subsampled to keep CPU experiments feasible)
    rng = random.Random(SEED)
    others = sorted(set(dtext) - test_pos)
    rng.shuffle(others)
    eval_ids = sorted(test_pos | set(others[:n_distractors]))

    # training pairs: never use a document that is a test positive (no leakage of test labels)
    pairs = [(qtext[q], dtext[d]) for q, ds in train.items() for d in ds if d not in test_pos and q in qtext]
    rng.shuffle(pairs)
    return dict(eval_docs={i: dtext[i] for i in eval_ids}, test_queries={q: qtext[q] for q in test if q in qtext},
                test_rel={q: ds for q, ds in test.items() if q in qtext}, train_pairs=pairs)
