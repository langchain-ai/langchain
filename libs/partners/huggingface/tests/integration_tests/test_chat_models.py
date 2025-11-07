from langchain_core.messages import AIMessageChunk

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


def test_stream_usage() -> None:
    """Test we are able to configure stream options on models that require it."""
    llm = HuggingFaceEndpoint(  # type: ignore[call-arg]  # (model is inferred in class)
        repo_id="google/gemma-3-27b-it",
        task="conversational",
        provider="nebius",
    )

    model = ChatHuggingFace(llm=llm, stream_usage=True)

    full: AIMessageChunk | None = None
    for chunk in model.stream("hello"):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk

    assert isinstance(full, AIMessageChunk)
    assert full.usage_metadata
