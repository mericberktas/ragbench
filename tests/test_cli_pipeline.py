"""The whole stage-2 path (sample -> generate -> judge -> report) with a fake transport instead of the API."""
import json

import ragbench.cli as cli
from ragbench.llm import LLMClient


def fake_data(tmp_path, n_articles=12):
    chunks, queries = [], []
    for a in range(n_articles):
        chunks.append({"id": f"a{a}#c0", "article_id": f"a{a}", "title": f"T{a}", "heading_path": "Lead", "text": f"bilgi {a}"})
        chunks.append({"id": f"a{a}#c1", "article_id": f"a{a}", "title": f"T{a}", "heading_path": "Body", "text": f"baska {a}"})
        for j in range(2):
            queries.append({"qid": f"a{a}#q{j}", "article_id": f"a{a}", "question": f"soru {a} {j}", "answer": f"bilgi {a}",
                            "category": "FACTUAL" if j == 0 else "INTERPRETATION", "relevant": [f"a{a}#c0"]})
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"meta": {}, "articles": [], "corpus": chunks, "queries": queries}), encoding="utf-8")
    return path, chunks, queries


def reply(messages):
    user = messages[-1]["content"]
    if messages[0]["content"].startswith("Bir değerlendirme"):
        ref = user.split("Referans cevap: ")[1].split("\n")[0]
        cand = user.split("Aday cevap: ")[1]
        return json.dumps({"verdict": "correct" if ref in cand else "incorrect", "reason": "x"})
    # generator: copies the first context chunk's text if there is a context, else says it does not know
    if "Bağlam:" in user:
        return user.split("[1] ")[1].split("\n")[0]
    return "Bilmiyorum."


def test_end_to_end(tmp_path, monkeypatch):
    data, chunks, queries = fake_data(tmp_path)
    monkeypatch.setattr(cli, "_client", lambda args: LLMClient(
        cache_path=tmp_path / "out" / "llm_cache.jsonl", max_calls=args.max_calls, max_workers=4,
        sleep=lambda s: None,
        transport=lambda model, messages, temperature, max_tokens, response_format: {
            "text": reply(messages), "prompt_tokens": 50, "completion_tokens": 5, "finish_reason": "stop"}))
    out = str(tmp_path / "out")
    qpath = str(tmp_path / "out" / "questions.json")
    cli.main(["sample", "--data", str(data), "--n", "10", "--out", qpath])
    questions = json.loads(open(qpath, encoding="utf-8").read())
    assert len(questions) == 10

    good = {q["qid"]: ["a" + q["article_id"][1:] + "#c0", "x"] for q in questions}       # gold chunk first
    bad = {q["qid"]: ["a" + q["article_id"][1:] + "#c1", "x"] for q in questions}        # wrong chunk first
    for name, res in (("good", good), ("bad", bad)):
        (tmp_path / f"{name}.json").write_text(json.dumps({"label": name, "k": 2, "results": res}), encoding="utf-8")

    common = ["--data", str(data), "--out-dir", out, "--questions", qpath]
    cli.main(["generate", *common, "--condition", "none", "--label", "no_retrieval"])
    cli.main(["generate", *common, "--condition", "gold", "--label", "gold_chunks"])
    for name in ("good", "bad"):
        cli.main(["generate", *common, "--condition", "retrieved", "--retrieval", str(tmp_path / f"{name}.json"),
                  "--label", name, "--k", "1"])
    for p in sorted((tmp_path / "out" / "answers").glob("*.jsonl")):
        cli.main(["judge", "--data", str(data), "--out-dir", out, "--answers", str(p)])
    cli.main(["report", "--data", str(data), "--out-dir", out, "--questions", qpath, "--reference", "gold"])

    def acc(stem):
        rows = [json.loads(l) for l in open(tmp_path / "out" / "verdicts" / f"{stem}.jsonl", encoding="utf-8")]
        return sum(r["verdict"] == "correct" for r in rows) / len(rows)

    assert acc("gold_chunks__gold") == 1.0 and acc("good__retrieved") == 1.0
    assert acc("bad__retrieved") == 0.0 and acc("no_retrieval__none") == 0.0
