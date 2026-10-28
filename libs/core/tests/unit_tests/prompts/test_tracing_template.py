"""Unit tests for prompt template observability and tracing metadata propagation."""

from typing import Any

from typing_extensions import override

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.prompt_values import ChatPromptValue, StringPromptValue
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
)


class CapturingCallbackHandler(BaseCallbackHandler):
    """Callback handler that captures options and metadata on LLM/chat model start."""

    def __init__(self) -> None:
        self.chat_model_options: dict[str, Any] | None = None
        self.llm_options: dict[str, Any] | None = None

    @override
    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.chat_model_options = options

    @override
    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.llm_options = options


def test_prompt_template_attaches_tracing_info() -> None:
    """Test that PromptTemplate attaches template_info on invoke and format_prompt."""
    prompt = PromptTemplate.from_template("Translate {text} into {language}")
    pv = prompt.invoke({"text": "hello", "language": "french"})

    assert isinstance(pv, StringPromptValue)
    assert pv.template_info is not None
    assert pv.template_info["template"] == "Translate {text} into {language}"
    assert "text" in pv.template_info["input_variables"]
    assert "language" in pv.template_info["input_variables"]
    assert pv.template_info["template_format"] == "f-string"

    # Also check format_prompt
    pv_fmt = prompt.format_prompt(text="hello", language="french")
    assert pv_fmt.template_info is not None
    assert pv_fmt.template_info["template"] == "Translate {text} into {language}"


def test_chat_prompt_template_attaches_tracing_info() -> None:
    """Test ChatPromptTemplate attaches template_info on invoke and format."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an assistant named {bot_name}."),
            ("human", "{user_message}"),
        ]
    )
    pv = prompt.invoke({"bot_name": "Antigravity", "user_message": "Hi"})

    assert isinstance(pv, ChatPromptValue)
    assert pv.template_info is not None
    assert "bot_name" in pv.template_info["input_variables"]
    assert "user_message" in pv.template_info["input_variables"]
    assert "messages" in pv.template_info
    assert len(pv.template_info["messages"]) == 2
    assert (
        pv.template_info["messages"][0]["template"]
        == "You are an assistant named {bot_name}."
    )
    assert pv.template_info["messages"][1]["template"] == "{user_message}"


def test_chat_model_receives_prompt_template_in_callbacks() -> None:
    """Test piped prompt | model forwards template to callback options."""
    handler = CapturingCallbackHandler()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("human", "What is {topic}?"),
        ]
    )
    model = FakeListChatModel(responses=["It is a cool thing."])
    chain = prompt | model

    chain.invoke({"topic": "LangChain"}, config={"callbacks": [handler]})

    assert handler.chat_model_options is not None
    assert "ls_prompt_template" in handler.chat_model_options
    ls_prompt = handler.chat_model_options["ls_prompt_template"]
    assert "topic" in ls_prompt["input_variables"]
    assert len(ls_prompt["messages"]) == 2


def test_llm_receives_prompt_template_in_callbacks() -> None:
    """Test that piped prompt | llm forwards ls_prompt_template to callback options."""
    handler = CapturingCallbackHandler()
    prompt = PromptTemplate.from_template("Question: {q}\nAnswer:")
    llm = FakeListLLM(responses=["42"])
    chain = prompt | llm

    chain.invoke({"q": "What is the meaning of life?"}, config={"callbacks": [handler]})

    assert handler.llm_options is not None
    assert "ls_prompt_template" in handler.llm_options
    ls_prompt = handler.llm_options["ls_prompt_template"]
    assert ls_prompt["template"] == "Question: {q}\nAnswer:"
    assert "q" in ls_prompt["input_variables"]


def test_prompt_value_pydantic_invariants_preserved() -> None:
    """Test that template_info does not pollute model_dump or affect equality."""
    p1 = PromptTemplate.from_template("Hello {name}")
    pv1 = p1.invoke({"name": "World"})
    pv2 = StringPromptValue(text="Hello World")

    # pv1 has template_info, pv2 does not
    assert pv1.template_info is not None
    assert pv2.template_info is None

    # But their string outputs and representations match
    assert pv1.to_string() == pv2.to_string()

    # Serialization/model_dump must not contain template_info
    dump = pv1.model_dump()
    assert "template_info" not in dump
    assert "_template_info" not in dump
