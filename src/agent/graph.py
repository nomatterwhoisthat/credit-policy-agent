"""LangGraph agent: llm -> tools -> llm ... -> numeric-grounding check -> end.

State machine
  call_llm   : ask the model (with tool specs)
  run_tools  : execute requested tool calls, append tool messages
  check      : if the model answered, verify that every number is grounded in tool outputs;
               on failure send one corrective message and loop back, otherwise finish
"""
from __future__ import annotations
import json
from typing import Any, TypedDict
from langgraph.graph import StateGraph, END
from .groundedness import ungrounded
from .tools import Toolbox, TOOL_SPECS

SYSTEM = ("You are a credit-risk assistant. Use tools for every client-specific number and for every policy statement. "
          "Never invent numbers. Explain declines with the reason codes returned by the scoring tool and never mention protected attributes.")
MAX_STEPS, MAX_REPAIRS = 8, 1


class State(TypedDict):
    messages: list[dict]
    steps: int
    repairs: int
    grounded: bool
    ungrounded_numbers: list[str]
    tool_trace: list[str]


def build_agent(llm, toolbox: Toolbox | None = None):
    tb = toolbox or Toolbox()

    def call_llm(s: State) -> dict[str, Any]:
        msg = llm.chat(s["messages"], TOOL_SPECS)
        return {"messages": s["messages"] + [msg], "steps": s["steps"] + 1}

    def run_tools(s: State) -> dict[str, Any]:
        msgs, trace = list(s["messages"]), list(s["tool_trace"])
        for call in s["messages"][-1]["tool_calls"]:
            name = call["function"]["name"]; args = json.loads(call["function"]["arguments"] or "{}")
            out = tb.run(name, args); trace.append(name)
            msgs.append({"role": "tool", "tool_call_id": call["id"], "name": name, "content": json.dumps(out)})
        return {"messages": msgs, "tool_trace": trace}

    def check(s: State) -> dict[str, Any]:
        answer = s["messages"][-1].get("content") or ""
        outs = [m["content"] for m in s["messages"] if m["role"] == "tool"]
        bad = ungrounded(answer, outs) if outs else []
        if bad and s["repairs"] < MAX_REPAIRS:
            fb = {"role": "user", "content": f"[guard] These numbers are not supported by tool results: {bad}. Rewrite the answer using only numbers from tool results."}
            return {"messages": s["messages"] + [fb], "repairs": s["repairs"] + 1, "grounded": False, "ungrounded_numbers": bad}
        return {"grounded": not bad, "ungrounded_numbers": bad}

    def after_llm(s: State) -> str:
        if s["steps"] >= MAX_STEPS:
            return "check"
        return "run_tools" if s["messages"][-1].get("tool_calls") else "check"

    def after_check(s: State) -> str:
        return "call_llm" if s["messages"][-1]["role"] == "user" else END

    g = StateGraph(State)
    g.add_node("call_llm", call_llm); g.add_node("run_tools", run_tools); g.add_node("check", check)
    g.set_entry_point("call_llm")
    g.add_conditional_edges("call_llm", after_llm, {"run_tools": "run_tools", "check": "check"})
    g.add_edge("run_tools", "call_llm")
    g.add_conditional_edges("check", after_check, {"call_llm": "call_llm", END: END})
    app = g.compile()

    def ask(question: str) -> dict:
        init: State = {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}],
                       "steps": 0, "repairs": 0, "grounded": True, "ungrounded_numbers": [], "tool_trace": []}
        final = app.invoke(init)
        return {"answer": final["messages"][-1].get("content"), "grounded": final["grounded"],
                "ungrounded_numbers": final["ungrounded_numbers"], "tools_used": final["tool_trace"], "steps": final["steps"],
                "repairs": final["repairs"]}
    ask.graph = app
    return ask
