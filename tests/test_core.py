import json

import numpy as np
import pytest

from ragbench.data import chunk_text, stratified_sample
from ragbench.generate import build_context, context_for, generate_answers
from ragbench.judge import judge_answers, parse_verdict
from ragbench.llm import BudgetExceeded, LLMClient
from ragbench.retrieve import gold_hit_rate, topk_ids
from ragbench.stats import cluster_bootstrap_mean, paired_cluster_bootstrap


class FakeTransport:
    def __init__(self, reply="ok", fail_times=0, fail_status=429):
        self.calls, self.reply, self.fail_times, self.fail_status = 0, reply, fail_times, fail_status

    def __call__(self, model, messages, temperature, max_tokens, response_format):
        self.calls += 1
        if self.calls <= self.fail_times:
            err = RuntimeError("boom")
            err.status_code = self.fail_status
            raise err
        text = self.reply(messages) if callable(self.reply) else self.reply
        return {"text": text, "prompt_tokens": 100, "completion_tokens": 10, "finish_reason": "stop"}


def make_client(transport, **kw):
    return LLMClient(transport=transport, sleep=lambda s: None, max_workers=4, **kw)


def questions(n_articles=6, per=2):
    out = []
    for a in range(n_articles):
        for j in range(per):
            out.append({"qid": f"a{a}#q{j}", "article_id": f"a{a}", "question": f"soru {a}{j}",
                        "answer": f"cevap {a}{j}", "category": "FACTUAL" if j == 0 else "INTERPRETATION",
                        "relevant": [f"a{a}#c0"]})
    return out


def test_cache_returns_same_text_and_skips_second_call(tmp_path):
    t = FakeTransport("hello")
    c = make_client(t, cache_path=tmp_path / "c.jsonl")
    m = [{"role": "user", "content": "x"}]
    assert c.complete("m", m)["text"] == "hello" and t.calls == 1
    assert c.complete("m", m)["cached"] and t.calls == 1
    c2 = make_client(FakeTransport("other"), cache_path=tmp_path / "c.jsonl")  # new process reads the file
    assert c2.complete("m", m)["text"] == "hello"


def test_cache_key_depends_on_every_request_field(tmp_path):
    t = FakeTransport()
    c = make_client(t, cache_path=tmp_path / "c.jsonl")
    m = [{"role": "user", "content": "x"}]
    c.complete("m", m); c.complete("m2", m); c.complete("m", m, temperature=0.5); c.complete("m", m, max_tokens=7)
    assert t.calls == 4


def test_retry_then_success_and_non_retryable_raises():
    t = FakeTransport(fail_times=2, fail_status=429)
    assert make_client(t).complete("m", [{"role": "user", "content": "x"}])["text"] == "ok" and t.calls == 3
    bad = FakeTransport(fail_times=1, fail_status=400)
    with pytest.raises(RuntimeError):
        make_client(bad).complete("m", [{"role": "user", "content": "x"}])
    assert bad.calls == 1


def test_call_cap_counts_live_calls_only():
    c = make_client(FakeTransport(), max_calls=2)
    a, b, d = ([{"role": "user", "content": s}] for s in "abd")
    c.complete("m", a); c.complete("m", b); c.complete("m", a)  # third is a cache hit
    with pytest.raises(BudgetExceeded):
        c.complete("m", d)


def test_cost_estimate_uses_known_prices():
    c = make_client(FakeTransport())
    c.complete("zai-org/GLM-5.3-Flash", [{"role": "user", "content": "x"}])
    assert c.cost_usd == pytest.approx((100 * 0.15 + 10 * 0.50) / 1e6)


def test_parse_verdict():
    assert parse_verdict('{"verdict": "correct", "reason": "ok"}') == ("correct", "ok")
    assert parse_verdict('```json\n{"verdict": "Abstain", "reason": ""}\n```')[0] == "abstain"
    assert parse_verdict("not json")[0] == "invalid"
    assert parse_verdict('{"verdict": "maybe"}')[0] == "invalid"


def test_judge_and_generate_round_trip():
    qs = questions(2, 1)
    gen = make_client(FakeTransport(lambda m: "cevap " + m[-1]["content"].split("soru ")[-1]))
    ctx = [context_for("none", q, {}) for q in qs]
    answers = generate_answers(gen, "m", qs, ctx)
    assert [a["qid"] for a in answers] == [q["qid"] for q in qs]
    jd = make_client(FakeTransport('{"verdict": "correct", "reason": "x"}'))
    rows = judge_answers(jd, "m", {q["qid"]: q for q in qs}, answers)
    assert all(r["verdict"] == "correct" for r in rows)


def test_contexts_per_condition():
    chunks = {"a0#c0": {"id": "a0#c0", "title": "T", "heading_path": "H", "text": "metin"},
              "a0#c1": {"id": "a0#c1", "title": "T", "heading_path": "H2", "text": "baska"}}
    q = {"qid": "a0#q0", "relevant": ["a0#c0"]}
    assert context_for("none", q, chunks) is None
    assert "metin" in context_for("gold", q, chunks) and "baska" not in context_for("gold", q, chunks)
    retrieved = {"a0#q0": ["a0#c1", "a0#c0"]}
    ctx = context_for("retrieved", q, chunks, retrieved, k=1)
    assert ctx.startswith("[1] T > H2") and "metin" not in ctx
    assert chunk_text(chunks["a0#c0"], "plain") == "metin"
    assert build_context(chunks, ["a0#c0", "a0#c1"]).count("[") == 2


def test_stratified_sample_one_per_article_and_deterministic():
    qs = questions(40, 2)
    a = stratified_sample(qs, 20, seed=1)
    assert len(a) == 20 and len({q["article_id"] for q in a}) == 20
    assert a == stratified_sample(qs, 20, seed=1) and a != stratified_sample(qs, 20, seed=2)
    assert len(stratified_sample(qs, 1000)) == 40


def test_topk_ids_matches_brute_force():
    rng = np.random.default_rng(0)
    d = rng.normal(size=(300, 16)).astype(np.float32); d /= np.linalg.norm(d, axis=1, keepdims=True)
    q = rng.normal(size=(50, 16)).astype(np.float32); q /= np.linalg.norm(q, axis=1, keepdims=True)
    idx, sc = topk_ids(q, d, 5, block=16)
    want = np.argsort(-(q @ d.T), axis=1)[:, :5]
    assert (idx == want).all() and (np.diff(sc, axis=1) <= 1e-6).all()


def test_gold_hit_rate():
    qs = questions(2, 1)
    assert gold_hit_rate({"a0#q0": ["a0#c0", "x"], "a1#q0": ["x", "y"]}, qs, 2) == 0.5


def test_bootstrap_interval_and_pairing():
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(200), 2)
    a = (rng.random(400) < 0.7).astype(float)
    b = (rng.random(400) < 0.4).astype(float)
    mean, (lo, hi) = cluster_bootstrap_mean(a, clusters, n_boot=2000)
    assert lo < mean < hi and abs(mean - a.mean()) < 1e-12
    d, (dlo, dhi), sig = paired_cluster_bootstrap(a, b, clusters, n_boot=2000)
    assert sig and dlo > 0 and d == pytest.approx(a.mean() - b.mean())
    _, _, sig_same = paired_cluster_bootstrap(a, a, clusters, n_boot=500)
    assert not sig_same
