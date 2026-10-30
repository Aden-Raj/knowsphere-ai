import os
import sys
import types

import pytest

from app.services.embeddings import SentenceTransformerEmbedder


def test_embedding_generation_uses_configured_cpu_device(monkeypatch) -> None:
    calls = {}

    class StubModel:
        def __init__(self, model_name, device, cache_folder=None):
            calls["model"] = (model_name, device, cache_folder)

        def encode(self, texts, **kwargs):
            calls["encode"] = (texts, kwargs)
            return [[0.6, 0.8] for _ in texts]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=StubModel),
    )
    embedder = SentenceTransformerEmbedder("test-model", device="cpu")

    vectors = embedder.embed(["one chunk", "another chunk"])

    assert vectors == [[0.6, 0.8], [0.6, 0.8]]
    assert calls["model"] == ("test-model", "cpu", None)
    assert calls["encode"][1]["device"] == "cpu"


@pytest.mark.real_model
@pytest.mark.skipif(os.getenv("RUN_REAL_EMBEDDING_TESTS") != "1", reason="opt-in model download/inference smoke test")
def test_real_sentence_transformer_embedding_smoke() -> None:
    vectors = SentenceTransformerEmbedder("all-MiniLM-L6-v2", device="cpu").embed(
        ["KnowSphere indexes document knowledge."]
    )

    assert len(vectors) == 1
    assert len(vectors[0]) == 384
    assert all(isinstance(value, float) for value in vectors[0])
