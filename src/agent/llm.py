"""LLM back-ends. Any OpenAI-compatible chat-completions endpoint with function calling works
(OpenAI, aimlapi.com, vLLM, Ollama, ...). `ScriptedLLM` is a deterministic offline policy used for tests and demos."""
from __future__ import annotations
import json, os, re, uuid
import requests


class OpenAICompatibleLLM:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None, timeout: int = 60):
        self.base = (base_url or os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.key = api_key or os.environ["LLM_API_KEY"]
        self.model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
        self.timeout = timeout

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        r = requests.post(f"{self.base}/chat/completions", timeout=self.timeout,
                          headers={"Authorization": f"Bearer {self.key}"},
                          json={"model": self.model, "messages": messages, "tools": tools, "temperature": 0})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]


def _call(name: str, **args) -> dict:
    return {"id": "call_" + uuid.uuid4().hex[:8], "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class ScriptedLLM:
    """Rule-based stand-in for an LLM. It routes by keywords, then writes the answer from tool outputs.
    `hallucinate_first=True` makes the first final answer contain an invented number to exercise the groundedness guard."""

    def __init__(self, hallucinate_first: bool = False):
        self.hallucinate_first, self._done_hallucination = hallucinate_first, False

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        user = next(m["content"] for m in reversed(messages) if m["role"] == "user" and not m["content"].startswith("[guard]"))
        tool_msgs = [m for m in messages if m["role"] == "tool"]
        if not tool_msgs:
            calls = []
            m = re.search(r"\b(low_risk|high_risk|borderline)\b", user)
            if m:
                calls.append(_call("score_applicant", sample_id=m.group(1)))
            if re.search(r"policy|why|threshold|psi|define|what is|explain|fair|monitor|break-even|rule", user, re.I):
                calls.append(_call("search_knowledge", query=user))
            return {"role": "assistant", "content": None, "tool_calls": calls or [_call("search_knowledge", query=user)]}
        data = [(m["name"], json.loads(m["content"])) for m in tool_msgs]
        parts = []
        for name, d in data:
            if name == "score_applicant":
                rs = "; ".join(f'{r["meaning"]} ({r["effect"]})' for r in d["reasons"])
                parts.append(f'Decision: {d["decision"]}. Calibrated PD is {d["pd"]:.4f} against a break-even PD of {d["breakeven_pd"]:.4f}, '
                             f'expected NPV {d["expected_npv"]:.1f}. Main drivers: {rs}.')
            else:
                top = d["results"][0]
                parts.append(f'From {top["source"]}: {top["text"][:300]}')
        text = " ".join(parts)
        if self.hallucinate_first and not self._done_hallucination:
            self._done_hallucination = True
            text += " The client also has a 71.3 % chance of recovering the full balance."
        return {"role": "assistant", "content": text}
