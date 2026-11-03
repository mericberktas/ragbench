"""Loading RAGTurk (the single-file derived copy kept on Drive) and drawing the question sample."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path


def load_ragturk(path: str | Path) -> tuple[list[dict], list[dict]]:
    """Returns (chunks, questions). Chunks: id, article_id, title, heading_path, text.
    Questions: qid, article_id, question, answer, category, relevant (gold chunk ids)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["corpus"], data["queries"]


def chunk_text(chunk: dict, doc_format: str = "title_path") -> str:
    if doc_format == "title_path":
        return f"{chunk['title']} > {chunk['heading_path']}\n{chunk['text']}"
    return chunk["text"]


def stratified_sample(questions: list[dict], n: int, seed: int = 42, key: str = "category") -> list[dict]:
    """At most one question per article (so the sample's units are independent), then proportional quotas
    per `key`. Deterministic for a given seed. Returns questions sorted by qid."""
    rng = random.Random(seed)
    by_article: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        by_article[q["article_id"]].append(q)
    pool = [rng.choice(sorted(qs, key=lambda q: q["qid"])) for _, qs in sorted(by_article.items())]
    if n >= len(pool):
        return sorted(pool, key=lambda q: q["qid"])
    groups: dict[str, list[dict]] = defaultdict(list)
    for q in pool:
        groups[q[key]].append(q)
    quotas = {g: int(n * len(qs) / len(pool)) for g, qs in groups.items()}
    # hand the rounding remainder to the largest groups
    for g in sorted(groups, key=lambda g: -len(groups[g]))[: n - sum(quotas.values())]:
        quotas[g] += 1
    picked: list[dict] = []
    for g, qs in sorted(groups.items()):
        qs = sorted(qs, key=lambda q: q["qid"])
        rng.shuffle(qs)
        picked.extend(qs[: quotas[g]])
    return sorted(picked, key=lambda q: q["qid"])
