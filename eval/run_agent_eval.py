"""Checks tool routing, decision consistency, content and numeric grounding for a list of cases.
Default backend is the offline ScriptedLLM (tests the graph, tools and guard, NOT language-model quality).
With LLM_API_KEY set, the same cases run against a real model: that is the intended use for model comparison."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agent.graph import build_agent
from src.agent.llm import OpenAICompatibleLLM, ScriptedLLM
from src.agent.tools import Toolbox
from src.agent.kb import KnowledgeBase


def main():
    llm = OpenAICompatibleLLM() if os.environ.get("LLM_API_KEY") else ScriptedLLM()
    tb = Toolbox(kb=KnowledgeBase("kb", use_dense=True))
    ask = build_agent(llm, tb)
    cases = json.load(open("eval/agent_cases.json")); rows = []
    for c in cases:
        r = ask(c["q"]); ans = (r["answer"] or "").lower()
        rows.append({"q": c["q"], "tools_ok": set(c["tools"]) <= set(r["tools_used"]), "grounded": r["grounded"],
                     "decision_ok": (c.get("decision") in ans) if c.get("decision") else None,
                     "content_ok": (c["must_contain"].lower() in ans) if c.get("must_contain") else None, "repairs": r["repairs"]})
    n = len(rows); summ = {"backend": type(llm).__name__, "cases": n,
                           "tool_routing": sum(r["tools_ok"] for r in rows) / n, "grounded": sum(r["grounded"] for r in rows) / n,
                           "decision_ok": [r["decision_ok"] for r in rows if r["decision_ok"] is not None],
                           "content_ok": [r["content_ok"] for r in rows if r["content_ok"] is not None]}
    json.dump({"summary": summ, "rows": rows}, open("results/agent_eval.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
