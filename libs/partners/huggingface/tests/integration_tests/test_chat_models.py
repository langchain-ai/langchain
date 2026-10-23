import os
from collections.abc import Callable

import torch
from langchain_core.messages import AIMessageChunk

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


def test_stream_usage() -> None:
    """Test we are able to configure stream options on models that require it."""
    llm = HuggingFaceEndpoint(  # type: ignore[call-arg]  # (model is inferred in class)
        repo_id="zai-org/GLM-5.2",
        task="conversational",
        provider="scaleway",
    )

    model = ChatHuggingFace(llm=llm, stream_usage=True)

    full: AIMessageChunk | None = None
    for chunk in model.stream("hello"):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk

    assert isinstance(full, AIMessageChunk)
    assert full.usage_metadata


def test_from_model_id_runs_on_accelerator(
    accelerator_device: int | str,
    assert_on_accelerator: Callable[[torch.device], None],
) -> None:
    """Chat with a local pipeline model on an accelerator, invoking and streaming."""
    model_id = os.getenv("HF_TEST_CHAT_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")

    model = ChatHuggingFace.from_model_id(
        model_id=model_id, device=accelerator_device, max_new_tokens=8
    )

    assert model.invoke("Hello").content
    assert "".join(str(chunk.content) for chunk in model.stream("Hello")).strip()
    assert_on_accelerator(model.llm.pipeline.model.device)
