import json
from src.agent.groundedness import ungrounded, allowed_numbers
from src.agent.kb import KnowledgeBase, load_chunks
from src.agent.tools import Toolbox
from src.agent.graph import build_agent
from src.agent.llm import ScriptedLLM

KB = KnowledgeBase("kb", use_dense=False)
TB = Toolbox(kb=KB)


def test_groundedness_accepts_rounded_and_percent_forms():
    out = [json.dumps({"pd": 0.0859, "breakeven_pd": 0.0844})]
    assert ungrounded("PD is 0.0859, i.e. 8.6 % and the break-even is 8.44 %.", out) == []


def test_groundedness_flags_invented_number():
    out = [json.dumps({"pd": 0.0859})]
    assert ungrounded("PD is 0.0859 and recovery chance is 71.3 %", out) == ["71.3"]


def test_chunks_have_sources():
    ch = load_chunks("kb")
    assert len(ch) >= 15 and all(c["source"].endswith(".md") for c in ch)


def test_kb_finds_psi_threshold():
    top = KB.search("what PSI value triggers a model investigation", k=2)
    assert any(r["source"] == "monitoring_policy.md" for r in top)


def test_score_tool_consistent_with_policy():
    hi = TB.run("score_applicant", {"sample_id": "high_risk"})
    lo = TB.run("score_applicant", {"sample_id": "low_risk"})
    assert hi["decision"] == "decline" and lo["decision"] == "approve"
    assert hi["pd"] > hi["breakeven_pd"] > 0 and lo["pd"] < lo["breakeven_pd"]
    assert len(hi["reasons"]) == 3 and all("SEX" not in r["feature"] and "MARRIAGE" not in r["feature"] for r in hi["reasons"])


def test_tool_errors_are_returned_not_raised():
    assert "error" in TB.run("score_applicant", {"client": {"AGE": 30}})
    assert "error" in TB.run("nope", {})


def test_agent_end_to_end_grounded():
    r = build_agent(ScriptedLLM(), TB)("Why is the borderline client declined? Explain the policy.")
    assert r["grounded"] and set(r["tools_used"]) == {"score_applicant", "search_knowledge"} and "decline" in r["answer"]


def test_agent_repairs_hallucination():
    r = build_agent(ScriptedLLM(hallucinate_first=True), TB)("Should the low_risk client be approved?")
    assert r["repairs"] == 1 and r["grounded"] and "71.3" not in r["answer"]
