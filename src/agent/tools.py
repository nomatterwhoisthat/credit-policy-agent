"""Agent tools: client scoring (PD, decision, reason codes) and knowledge-base search."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "score_applicant",
        "description": "Score a credit-card client: calibrated probability of default, approve/decline decision by expected NPV, break-even PD and top-3 SHAP reason codes.",
        "parameters": {"type": "object", "properties": {
            "sample_id": {"type": "string", "description": "id of a bundled sample client: low_risk, high_risk or borderline"},
            "client": {"type": "object", "description": "raw client record with LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_1..PAY_6, BILL_AMT1..6, PAY_AMT1..6"}}}}},
    {"type": "function", "function": {
        "name": "search_knowledge",
        "description": "Search the internal knowledge base (model card, decision policy, glossary, monitoring and fairness policies, feature dictionary).",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]


class Toolbox:
    def __init__(self, kb=None, scorer=None, samples_path: Path | None = None):
        self._kb, self._scorer = kb, scorer
        self.samples = json.load(open(samples_path or ROOT / "examples" / "sample_clients.json"))

    @property
    def kb(self):
        if self._kb is None:
            from .kb import KnowledgeBase
            self._kb = KnowledgeBase(ROOT / "kb")
        return self._kb

    @property
    def scorer(self):
        if self._scorer is None:
            from ..credit.scoring import get_scorer
            self._scorer = get_scorer()
        return self._scorer

    def run(self, name: str, args: dict) -> dict:
        try:
            if name == "score_applicant":
                client = args.get("client") or {k: v for k, v in self.samples[args["sample_id"]].items() if not k.startswith("_")}
                return self.scorer.score(client)
            if name == "search_knowledge":
                return {"results": self.kb.search(args["query"], k=3)}
            return {"error": f"unknown tool {name}"}
        except Exception as e:                      # tool errors are returned to the model, never raised
            return {"error": f"{type(e).__name__}: {e}"}
