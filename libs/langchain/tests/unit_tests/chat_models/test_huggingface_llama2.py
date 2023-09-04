"""Test Hugging Face Llama-2 Chat model."""

from langchain.chat_models.huggingface_llama2 import ChatLlama2Hf
from langchain.schema.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)


def test_format_messages_as_text_with_system() -> None:
    messages = [
        SystemMessage(content="System Prompt."),
        HumanMessage(content="Human Message."),
        AIMessage(content="AI response."),
        HumanMessage(content="Second Human Message."),
    ]

    ground_truth = "<s>[INST] <<SYS>>\nSystem Prompt.\n<</SYS>>\n\nHuman Message. [/INST] AI response. </s><s>[INST] Second Human Message. [/INST] "  # noqa: E501

    messages_as_str = ChatLlama2Hf.format_messages_as_text(messages=messages)
    assert messages_as_str == ground_truth


def test_format_messages_as_text_without_system() -> None:
    messages = [
        HumanMessage(content="Human Message."),
        AIMessage(content="AI response."),
        HumanMessage(content="Second Human Message."),
        AIMessage(content="Second AI response."),
    ]

    ground_truth = "<s>[INST] Human Message. [/INST] AI response. </s><s>[INST] Second Human Message. [/INST] Second AI response. </s><s>[INST] "  # noqa: E501

    messages_as_str = ChatLlama2Hf.format_messages_as_text(messages=messages)
    assert messages_as_str == ground_truth
