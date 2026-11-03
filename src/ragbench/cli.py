"""Stage 2 and reporting (no GPU, only API calls):

  python -m ragbench.cli sample    --data D.json --n 500 --out out/questions.json
  python -m ragbench.cli smoke     --n 3                       # three live calls: is the key and the model alive?
  python -m ragbench.cli generate  --data D.json --questions out/questions.json --condition retrieved \
                                   --retrieval out/retrieval/<label>.json --label <label> --out-dir out
  python -m ragbench.cli judge     --data D.json --answers out/answers/<label>__retrieved.jsonl --out-dir out
  python -m ragbench.cli report    --data D.json --questions out/questions.json --out-dir out --reference <label>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import load_ragturk, stratified_sample
from .generate import context_for, generate_answers, read_jsonl, write_jsonl
from .judge import judge_answers
from .llm import LLMClient
from .stats import cluster_bootstrap_mean, paired_cluster_bootstrap

GEN_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
JUDGE_MODEL = "zai-org/GLM-5.3-Flash"


def _client(args) -> LLMClient:
    return LLMClient(cache_path=Path(args.out_dir) / "llm_cache.jsonl", max_calls=args.max_calls,
                     max_workers=args.workers)


def cmd_sample(args):
    _, questions = load_ragturk(args.data)
    picked = stratified_sample(questions, args.n, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(picked, ensure_ascii=False), encoding="utf-8")
    cats = {}
    for q in picked:
        cats[q["category"]] = cats.get(q["category"], 0) + 1
    print(f"{len(picked)} questions from {len({q['article_id'] for q in picked})} articles; {cats}")


def cmd_smoke(args):
    client = LLMClient(max_calls=2 * args.n, max_workers=2)
    msgs = [[{"role": "user", "content": "Türkiye'nin başkenti neresidir? Tek kelimeyle yanıtla."}]]
    for model in (GEN_MODEL, JUDGE_MODEL):
        for _ in range(args.n):
            out = client.complete(model, msgs[0], max_tokens=64)
            print(f"{model}: {out['text']!r} (finish={out.get('finish_reason')}, "
                  f"{out['prompt_tokens']} in / {out['completion_tokens']} out)")
    print(client.summary())


def cmd_generate(args):
    chunks, _ = load_ragturk(args.data)
    chunks_by_id = {c["id"]: c for c in chunks}
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    retrieved = json.loads(Path(args.retrieval).read_text(encoding="utf-8"))["results"] if args.retrieval else None
    label = args.label or args.condition
    contexts = [context_for(args.condition, q, chunks_by_id, retrieved, args.k, args.doc_format) for q in questions]
    client = _client(args)
    rows = generate_answers(client, args.gen_model, questions, contexts)
    for r in rows:
        r.update(label=label, condition=args.condition)
    out = Path(args.out_dir) / "answers" / f"{label}__{args.condition}.jsonl"
    write_jsonl(out, rows)
    truncated = sum(r["finish_reason"] == "length" for r in rows)
    print(f"{out}: {len(rows)} answers ({truncated} cut off by max_tokens); {client.summary()}")


def cmd_judge(args):
    _, all_questions = load_ragturk(args.data)
    by_qid = {q["qid"]: q for q in all_questions}
    answers = read_jsonl(args.answers)
    client = _client(args)
    rows = judge_answers(client, args.judge_model, by_qid, answers)
    name = Path(args.answers).stem
    out = Path(args.out_dir) / "verdicts" / f"{name}.jsonl"
    write_jsonl(out, rows)
    counts = {v: sum(r["verdict"] == v for r in rows) for v in ("correct", "incorrect", "abstain", "invalid")}
    cut = sum(r.get("finish_reason") == "length" for r in rows)
    print(f"{out}: {counts} ({cut} replies hit max_tokens); {client.summary()}")


def cmd_report(args):
    _, all_questions = load_ragturk(args.data)
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    qids = [q["qid"] for q in questions]
    article = np.array([q["article_id"] for q in questions])
    cat = np.array([q["category"] for q in questions])
    scores = {}
    for path in sorted((Path(args.out_dir) / "verdicts").glob("*.jsonl")):
        rows = {r["qid"]: r["verdict"] for r in read_jsonl(path)}
        if set(rows) != set(qids):
            print(f"skipping {path.name}: questions differ from the sample")
            continue
        scores[path.stem] = {"acc": np.array([rows[i] == "correct" for i in qids], dtype=float),
                             "abstain": np.array([rows[i] == "abstain" for i in qids], dtype=float),
                             "invalid": np.array([rows[i] == "invalid" for i in qids], dtype=float)}
    ref = next((k for k in scores if k.startswith(args.reference)), None) if args.reference else None
    print(f"{'system':<46}{'accuracy':>9}  {'95% interval':<16}{'abstain':>8}{'invalid':>8}" + ("   diff vs reference" if ref else ""))
    for name in sorted(scores, key=lambda n: -scores[n]["acc"].mean()):
        s = scores[name]
        mean, (lo, hi) = cluster_bootstrap_mean(s["acc"], article)
        line = f"{name:<46}{mean:>9.3f}  [{lo:.3f}, {hi:.3f}]  {s['abstain'].mean():>8.3f}{s['invalid'].mean():>8.3f}"
        if ref and name != ref:
            d, (dlo, dhi), sig = paired_cluster_bootstrap(s["acc"], scores[ref]["acc"], article)
            line += f"   {d:+.3f} [{dlo:+.3f}, {dhi:+.3f}]{' *' if sig else ''}"
        print(line)
    print("(* = the paired interval excludes 0; intervals resample articles)")
    for c in sorted(set(cat)):
        m = cat == c
        print(f"\n{c} ({m.sum()} questions)")
        for name in sorted(scores, key=lambda n: -scores[n]["acc"][m].mean()):
            print(f"  {name:<44}{scores[name]['acc'][m].mean():>7.3f}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, api=True):
        sp.add_argument("--data", required=True)
        sp.add_argument("--out-dir", default="out")
        if api:
            sp.add_argument("--max-calls", type=int, default=2000, help="hard cap on live API calls per command")
            sp.add_argument("--workers", type=int, default=16)

    s = sub.add_parser("sample"); common(s, api=False)
    s.add_argument("--n", type=int, default=500); s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", required=True); s.set_defaults(fn=cmd_sample)

    s = sub.add_parser("smoke"); s.add_argument("--n", type=int, default=3); s.set_defaults(fn=cmd_smoke)

    s = sub.add_parser("generate"); common(s)
    s.add_argument("--questions", required=True)
    s.add_argument("--condition", choices=("retrieved", "gold", "none"), required=True)
    s.add_argument("--retrieval"); s.add_argument("--label"); s.add_argument("--k", type=int, default=5)
    s.add_argument("--doc-format", choices=("content", "title_path"), default="content",
                   help="'content' = chunk text only (as in the retrieval checks); 'title_path' adds title and heading")
    s.add_argument("--gen-model", default=GEN_MODEL); s.set_defaults(fn=cmd_generate)

    s = sub.add_parser("judge"); common(s)
    s.add_argument("--answers", required=True); s.add_argument("--judge-model", default=JUDGE_MODEL)
    s.set_defaults(fn=cmd_judge)

    s = sub.add_parser("report"); common(s, api=False)
    s.add_argument("--questions", required=True); s.add_argument("--reference", default=None)
    s.set_defaults(fn=cmd_report)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
