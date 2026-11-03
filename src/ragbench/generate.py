"""Answer generation: one answer per (system, question) with a condition-specific context."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

from .data import chunk_text
from .llm import LLMClient
from .prompts import generation_messages


def build_context(chunks_by_id: dict[str, dict], chunk_ids: list[str], doc_format: str = "title_path") -> str:
    return "\n\n".join(f"[{i + 1}] {chunk_text(chunks_by_id[c], doc_format)}" for i, c in enumerate(chunk_ids))


def context_for(condition: str, question: dict, chunks_by_id: dict[str, dict], retrieved: Optional[dict] = None,
                k: int = 5, doc_format: str = "title_path") -> Optional[str]:
    """'retrieved': the model's top-k chunks; 'gold': the chunks the question was written from;
    'none': no context (the generator's own knowledge)."""
    if condition == "none":
        return None
    if condition == "gold":
        return build_context(chunks_by_id, question["relevant"], doc_format)
    if condition == "retrieved":
        return build_context(chunks_by_id, retrieved[question["qid"]][:k], doc_format)
    raise ValueError(f"unknown condition {condition!r}")


def generate_answers(client: LLMClient, model: str, questions: list[dict], contexts: list[Optional[str]],
                     max_tokens: int = 300) -> list[dict]:
    messages = [generation_messages(q["question"], ctx) for q, ctx in zip(questions, contexts)]
    outs = client.complete_many(model, messages, temperature=0.0, max_tokens=max_tokens)
    return [{"qid": q["qid"], "answer": o["text"].strip(), "finish_reason": o.get("finish_reason")}
            for q, o in zip(questions, outs)]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
