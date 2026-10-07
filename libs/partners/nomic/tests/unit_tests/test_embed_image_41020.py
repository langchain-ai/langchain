"""Regression tests for issue #41020: ``embed_image`` and ``vision_model``."""

from unittest.mock import patch

from langchain_nomic.embeddings import NomicEmbeddings

_PATCH_TARGET = "langchain_nomic.embeddings.embed.image"


def _fake_result() -> dict:
    return {"embeddings": [[0.1, 0.2]]}


def test_embed_image_without_vision_model_does_not_send_model_none() -> None:
    """When ``vision_model`` is unset, ``model`` must not be forwarded as None.

    Forwarding ``model=None`` makes the Nomic SDK raise a bare ``AssertionError``
    instead of applying its documented default vision model.
    """
    emb = NomicEmbeddings(model="nomic-embed-text-v1.5")
    with patch(_PATCH_TARGET, return_value=_fake_result()) as mock_image:
        result = emb.embed_image(["https://example.com/cat.jpg"])

    assert result == [[0.1, 0.2]]
    _, kwargs = mock_image.call_args
    assert kwargs.get("model", "UNSET") is not None, (
        f"model=None was forwarded to the SDK: {kwargs!r}"
    )


def test_embed_image_forwards_explicit_vision_model() -> None:
    """An explicit ``vision_model`` is forwarded unchanged."""
    emb = NomicEmbeddings(
        model="nomic-embed-text-v1.5",
        vision_model="nomic-embed-vision-v1.5",
    )
    with patch(_PATCH_TARGET, return_value=_fake_result()) as mock_image:
        emb.embed_image(["https://example.com/cat.jpg"])

    _, kwargs = mock_image.call_args
    assert kwargs.get("model") == "nomic-embed-vision-v1.5"
