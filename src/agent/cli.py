"""python -m src.agent.cli "Why was the high_risk client declined and what is the policy on explanations?"
Set LLM_API_KEY (and optionally LLM_BASE_URL, LLM_MODEL) to use a real LLM; otherwise the offline ScriptedLLM runs."""
from __future__ import annotations
import json, os, sys
from .graph import build_agent
from .llm import OpenAICompatibleLLM, ScriptedLLM


def main():
    q = " ".join(sys.argv[1:]) or "Should the high_risk client be approved and why?"
    llm = OpenAICompatibleLLM() if os.environ.get("LLM_API_KEY") else ScriptedLLM()
    print("LLM backend:", type(llm).__name__)
    print(json.dumps(build_agent(llm)(q), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
