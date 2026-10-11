"""Deterministic numeric-grounding guard: every number in the final answer must be traceable to a tool result.
This catches the most damaging failure in a credit setting, an invented figure, without a second LLM call."""
from __future__ import annotations
import json, re

_NUM = re.compile(r"(?<![\w.])-?\d+(?:[.,]\d+)?")
IGNORE_SMALL_INTS = {str(i) for i in range(0, 11)}      # 'top 3', '6 months', ordinals


def _norm(x: str) -> str:
    x = x.replace(",", ".")
    return x.rstrip("0").rstrip(".") if "." in x else x


def numbers_in(text: str) -> set[str]:
    return {_norm(m.group()) for m in _NUM.finditer(text)}


def allowed_numbers(tool_outputs: list[str]) -> set[str]:
    allowed: set[str] = set()

    def add_float(v: float):
        for nd in range(0, 5):
            allowed.add(_norm(f"{round(v, nd):.{nd}f}"))
            allowed.add(_norm(f"{round(v * 100, nd):.{nd}f}"))       # percentages

    def walk(o):
        if isinstance(o, dict):
            for v in o.values(): walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
        elif isinstance(o, bool):
            return
        elif isinstance(o, (int, float)):
            add_float(float(o))
        elif isinstance(o, str):
            allowed.update(numbers_in(o))

    for t in tool_outputs:
        try:
            walk(json.loads(t))
        except json.JSONDecodeError:
            allowed.update(numbers_in(t))
    return allowed


def ungrounded(answer: str, tool_outputs: list[str]) -> list[str]:
    ok = allowed_numbers(tool_outputs)
    return sorted(n for n in numbers_in(answer) if n not in ok and n not in IGNORE_SMALL_INTS)
