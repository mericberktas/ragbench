"""Stage 1 (needs a GPU for the big models): encode chunks and questions, write each question's top-k chunk ids.

A model spec is a dict: `path` (a local folder) or `hf` + `rev` (a Hugging Face repo at a pinned revision), plus
optional `q` / `d` prefixes for queries / documents. Vectors are L2-normalised, so cosine = dot product.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .data import chunk_text


def topk_ids(q: np.ndarray, d: np.ndarray, k: int, block: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k by dot product, blockwise over the queries. Returns (indices, scores), best first."""
    idx = np.zeros((len(q), k), dtype=np.int64)
    sc = np.zeros((len(q), k), dtype=np.float32)
    for s in range(0, len(q), block):
        sims = q[s:s + block] @ d.T
        part = np.argpartition(-sims, kth=min(k, sims.shape[1] - 1), axis=1)[:, :k]
        part_sc = np.take_along_axis(sims, part, axis=1)
        order = np.argsort(-part_sc, axis=1)
        idx[s:s + block] = np.take_along_axis(part, order, axis=1)
        sc[s:s + block] = np.take_along_axis(part_sc, order, axis=1)
    return idx, sc


def gold_hit_rate(retrieved: dict[str, list[str]], questions: list[dict], k: int) -> float:
    """Share of questions with at least one gold chunk in the top-k (a sanity number, not a headline metric)."""
    hits = [bool(set(retrieved[q["qid"]][:k]) & set(q["relevant"])) for q in questions]
    return float(np.mean(hits))


def encode(model, texts: list[str], prefix: str, batch_size: int = 64) -> np.ndarray:
    vecs = model.encode([prefix + t for t in texts], batch_size=batch_size, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False)
    return vecs.astype(np.float32)


def retrieve_for_model(label: str, spec: dict, chunks: list[dict], questions: list[dict], out_dir: str | Path,
                       k: int = 5, doc_format: str = "content", model_dir: Optional[str] = None,
                       batch_size: int = 64, max_seq_length: int = 512) -> dict:
    """Writes <out_dir>/<label>.json = {"label", "k", "gold_hit", "results": {qid: [chunk ids]}} and returns it.
    Skips (and reads back) a finished result."""
    out_dir = Path(out_dir)
    out_path = out_dir / f"{label}.json"
    if out_path.exists():
        return json.loads(out_path.read_text(encoding="utf-8"))
    import torch
    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    if "path" in spec:
        source = spec["path"]
    else:
        from huggingface_hub import snapshot_download
        source = snapshot_download(spec["hf"], revision=spec["rev"], local_dir=model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(source, device=device)
    model.max_seq_length = max_seq_length
    if device == "cuda":
        model.half()
    d = encode(model, [chunk_text(c, doc_format) for c in chunks], spec.get("d", ""), batch_size)
    q = encode(model, [x["question"] for x in questions], spec.get("q", ""), batch_size)
    idx, _ = topk_ids(q, d, k)
    results = {x["qid"]: [chunks[j]["id"] for j in row] for x, row in zip(questions, idx)}
    out = {"label": label, "k": k, "doc_format": doc_format, "spec": {a: b for a, b in spec.items() if a != "path"},
           "gold_hit": gold_hit_rate(results, questions, k), "seconds": round(time.time() - t0, 1), "results": results}
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    tmp.replace(out_path)
    del model, d, q
    if device == "cuda":
        torch.cuda.empty_cache()
    return out
