"""Knowledge base over markdown documents: heading-level chunks, hybrid retrieval (BM25 + dense, RRF)."""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np

try:
    from ..retrievers import BM25, Dense, rrf
except ImportError:   # pragma: no cover
    from src.retrievers import BM25, Dense, rrf


def load_chunks(kb_dir: str | Path) -> list[dict]:
    chunks = []
    for p in sorted(Path(kb_dir).glob("*.md")):
        title, section, buf = "", "", []

        def flush():
            if buf and "".join(buf).strip():
                chunks.append({"id": f"{p.stem}#{len(chunks)}", "source": p.name, "title": title, "section": section,
                               "text": " ".join(x.strip() for x in buf if x.strip())})
        for line in p.read_text().splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
            elif line.startswith("## "):
                flush(); buf = []; section = line[3:].strip()
            else:
                buf.append(line)
        flush()
    return chunks


class KnowledgeBase:
    def __init__(self, kb_dir: str | Path, use_dense: bool = True, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.chunks = load_chunks(kb_dir)
        ids = [c["id"] for c in self.chunks]
        texts = [f'{c["title"]} - {c["section"]}. {c["text"]}' for c in self.chunks]
        self.by_id = {c["id"]: c for c in self.chunks}
        self.bm25 = BM25(ids, texts)
        self.dense = None
        if use_dense:
            from sentence_transformers import SentenceTransformer
            self.dense = Dense(SentenceTransformer(model_name, device="cpu"), ids, texts)

    def search(self, query: str, k: int = 3) -> list[dict]:
        rankings = [[d for d, _ in self.bm25.search(query, 10)]]
        if self.dense is not None:
            rankings.append([d for d, _ in self.dense.search_many([query], 10)[0]])
        ids = rrf(rankings)[:k]
        return [{"source": self.by_id[i]["source"], "section": self.by_id[i]["section"], "text": self.by_id[i]["text"]} for i in ids]
