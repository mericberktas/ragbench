"""LLM judging of generated answers against the reference answer (JSON verdicts)."""
from __future__ import annotations

import json
import re

from .llm import LLMClient
from .prompts import judge_messages

VERDICTS = ("correct", "incorrect", "abstain")


def parse_verdict(text: str) -> tuple[str, str]:
    """Returns (verdict, reason). A reply that is not valid JSON, or has an unknown verdict, becomes 'invalid'
    (kept, never silently counted as wrong)."""
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            obj = json.loads(match.group(0))
            verdict = str(obj.get("verdict", "")).strip().lower()
            if verdict in VERDICTS:
                return verdict, str(obj.get("reason", "")).strip()
        except json.JSONDecodeError:
            pass
    return "invalid", text.strip()[:200]


def judge_answers(client: LLMClient, model: str, questions_by_qid: dict[str, dict], answers: list[dict],
                  max_tokens: int = 200) -> list[dict]:
    messages = [judge_messages(questions_by_qid[a["qid"]]["question"], questions_by_qid[a["qid"]]["answer"], a["answer"])
                for a in answers]
    outs = client.complete_many(model, messages, temperature=0.0, max_tokens=max_tokens,
                                response_format={"type": "json_object"})
    rows = []
    for a, o in zip(answers, outs):
        verdict, reason = parse_verdict(o["text"])
        rows.append({"qid": a["qid"], "verdict": verdict, "reason": reason})
    return rows
