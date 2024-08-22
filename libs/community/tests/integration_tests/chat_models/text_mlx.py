"""Test MLX Chat Model."""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from langchain_community.chat_models.mlx import ChatMLX
from langchain_community.llms.mlx_pipeline import MLXPipeline


def test_default_call() -> None:
    """Test default model call."""
    llm = MLXPipeline.from_model_id(
        model_id="mlx-community/quantized-gemma-2b-it",
        pipeline_kwargs={"max_new_tokens": 10},
    )
    chat = ChatMLX(llm=llm)
    response = chat.invoke(input=[HumanMessage(content="Hello")])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_multiple_history() -> None:
    """Tests multiple history works."""
    llm = MLXPipeline.from_model_id(
        model_id="mlx-community/quantized-gemma-2b-it",
        pipeline_kwargs={"max_new_tokens": 10},
    )
    chat = ChatMLX(llm=llm)

    response = chat.invoke(
        input=[
            HumanMessage(content="Hello."),
            AIMessage(content="Hello!"),
            HumanMessage(content="How are you doing?"),
        ]
    )
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)
