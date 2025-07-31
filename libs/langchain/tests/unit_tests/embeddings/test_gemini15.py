# libs/langchain/tests/unit_tests/embeddings/test_gemini15.py
import types
from typing import Any

import pytest

from langchain.embeddings.gemini15 import Gemini15Embeddings


def fake_embed_content(*args: Any, **kwargs: Any) -> dict[str, list[list[float]]]:
    # استخرج النصوص سواءً أُرسلت positional أو keyword
    texts = args[0] if args else kwargs.get("input", [])
    # أرجِع متجهات وهميّة بطول 768
    return {"embedding": [[0.0] * 768 for _ in texts]}


@pytest.fixture(autouse=True)
def patch_genai(monkeypatch: pytest.MonkeyPatch) -> None:
    # إنشاء كائن genai.GenerativeModel وهمي
    fake_model = types.SimpleNamespace(embed_content=fake_embed_content)
    monkeypatch.setattr(
        "langchain.embeddings.gemini15.genai.GenerativeModel",
        lambda _: fake_model,
    )


def test_embed_query() -> None:
    emb = Gemini15Embeddings(api_key="fake")
    vec = emb.embed_query("مرحبا")
    assert len(vec) == 768
