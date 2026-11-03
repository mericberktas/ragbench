import json

import numpy as np
import pytest

pytest.importorskip("sentence_transformers")
pytest.importorskip("torch")


class FakeST:
    """Bag-of-characters 'embedding': texts sharing characters are close. Records the texts it was asked to encode."""
    seen: list = []

    def __init__(self, source, device=None):
        self.max_seq_length = None

    def half(self):
        return self

    def encode(self, texts, batch_size, normalize_embeddings, convert_to_numpy, show_progress_bar):
        FakeST.seen.extend(texts)
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for ch in t:
                out[i, ord(ch) % 64] += 1
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def test_retrieve_for_model_writes_and_skips(tmp_path, monkeypatch):
    import sentence_transformers
    from ragbench.retrieve import retrieve_for_model

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeST)
    chunks = [{"id": f"a{i}#c0", "article_id": f"a{i}", "title": "T", "heading_path": "H", "text": t}
              for i, t in enumerate(["kedi bahcede uyuyor", "roket aya gitti", "deniz mavi ve derin"])]
    questions = [{"qid": "a1#q0", "article_id": "a1", "question": "roket aya gitti mi", "answer": "x", "category": "F",
                  "relevant": ["a1#c0"]}]
    spec = {"path": "ignored", "q": "query: ", "d": "passage: "}
    out = retrieve_for_model("fake", spec, chunks, questions, tmp_path, k=2)
    assert out["results"]["a1#q0"][0] == "a1#c0" and out["gold_hit"] == 1.0
    assert "passage: roket aya gitti" in FakeST.seen and "query: roket aya gitti mi" in FakeST.seen
    saved = json.loads((tmp_path / "fake.json").read_text(encoding="utf-8"))
    assert saved["results"] == out["results"] and "path" not in saved["spec"]

    FakeST.seen.clear()
    again = retrieve_for_model("fake", spec, chunks, questions, tmp_path, k=2)   # finished: read back, no encoding
    assert again["results"] == out["results"] and FakeST.seen == []
