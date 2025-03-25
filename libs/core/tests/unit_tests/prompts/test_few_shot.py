"""Test few shot prompt template."""

from collections.abc import Sequence
from typing import Any

import pytest

from langchain_core.example_selectors import BaseExampleSelector
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.prompts.chat import SystemMessagePromptTemplate
from langchain_core.prompts.few_shot import (
    FewShotChatMessagePromptTemplate,
    FewShotPromptTemplate,
)
from langchain_core.prompts.prompt import PromptTemplate

EXAMPLE_PROMPT = PromptTemplate(
    input_variables=["question", "answer"], template="{question}: {answer}"
)


@pytest.fixture()
@pytest.mark.requires("jinja2")
def example_jinja2_prompt() -> tuple[PromptTemplate, list[dict[str, str]]]:
    example_template = "{{ word }}: {{ antonym }}"

    examples = [
        {"word": "happy", "antonym": "sad"},
        {"word": "tall", "antonym": "short"},
    ]

    return (
        PromptTemplate(
            input_variables=["word", "antonym"],
            template=example_template,
            template_format="jinja2",
        ),
        examples,
    )


def test_suffix_only() -> None:
    """Test prompt works with just a suffix."""
    suffix = "This is a {foo} test."
    input_variables = ["foo"]
    prompt = FewShotPromptTemplate(
        input_variables=input_variables,
        suffix=suffix,
        examples=[],
        example_prompt=EXAMPLE_PROMPT,
    )
    output = prompt.format(foo="bar")
    expected_output = "This is a bar test."
    assert output == expected_output


def test_auto_infer_input_variables() -> None:
    """Test prompt works with just a suffix."""
    suffix = "This is a {foo} test."
    prompt = FewShotPromptTemplate(
        suffix=suffix,
        examples=[],
        example_prompt=EXAMPLE_PROMPT,
    )
    assert prompt.input_variables == ["foo"]


def test_prompt_missing_input_variables() -> None:
    """Test error is raised when input variables are not provided."""
    # Test when missing in suffix
    template = "This is a {foo} test."
    with pytest.raises(ValueError):
        FewShotPromptTemplate(
            input_variables=[],
            suffix=template,
            examples=[],
            example_prompt=EXAMPLE_PROMPT,
            validate_template=True,
        )
    assert FewShotPromptTemplate(
        input_variables=[],
        suffix=template,
        examples=[],
        example_prompt=EXAMPLE_PROMPT,
    ).input_variables == ["foo"]

    # Test when missing in prefix
    template = "This is a {foo} test."
    with pytest.raises(ValueError):
        FewShotPromptTemplate(
            input_variables=[],
            suffix="foo",
            examples=[],
            prefix=template,
            example_prompt=EXAMPLE_PROMPT,
            validate_template=True,
        )
    assert FewShotPromptTemplate(
        input_variables=[],
        suffix="foo",
        examples=[],
        prefix=template,
        example_prompt=EXAMPLE_PROMPT,
    ).input_variables == ["foo"]


async def test_few_shot_functionality() -> None:
    """Test that few shot works with examples."""
    prefix = "This is a test about {content}."
    suffix = "Now you try to talk about {new_content}."
    examples = [
        {"question": "foo", "answer": "bar"},
        {"question": "baz", "answer": "foo"},
    ]
    prompt = FewShotPromptTemplate(
        suffix=suffix,
        prefix=prefix,
        input_variables=["content", "new_content"],
        examples=examples,
        example_prompt=EXAMPLE_PROMPT,
        example_separator="\n",
    )
    expected_output = (
        "This is a test about animals.\n"
        "foo: bar\n"
        "baz: foo\n"
        "Now you try to talk about party."
    )
    output = prompt.format(content="animals", new_content="party")
    assert output == expected_output
    output = await prompt.aformat(content="animals", new_content="party")
    assert output == expected_output


def test_partial_init_string() -> None:
    """Test prompt can be initialized with partial variables."""
    prefix = "This is a test about {content}."
    suffix = "Now you try to talk about {new_content}."
    examples = [
        {"question": "foo", "answer": "bar"},
        {"question": "baz", "answer": "foo"},
    ]
    prompt = FewShotPromptTemplate(
        suffix=suffix,
        prefix=prefix,
        input_variables=["new_content"],
        partial_variables={"content": "animals"},
        examples=examples,
        example_prompt=EXAMPLE_PROMPT,
        example_separator="\n",
    )
    output = prompt.format(new_content="party")
    expected_output = (
        "This is a test about animals.\n"
        "foo: bar\n"
        "baz: foo\n"
        "Now you try to talk about party."
    )
    assert output == expected_output


def test_partial_init_func() -> None:
    """Test prompt can be initialized with partial variables."""
    prefix = "This is a test about {content}."
    suffix = "Now you try to talk about {new_content}."
    examples = [
        {"question": "foo", "answer": "bar"},
        {"question": "baz", "answer": "foo"},
    ]
    prompt = FewShotPromptTemplate(
        suffix=suffix,
        prefix=prefix,
        input_variables=["new_content"],
        partial_variables={"content": lambda: "animals"},
        examples=examples,
        example_prompt=EXAMPLE_PROMPT,
        example_separator="\n",
    )
    output = prompt.format(new_content="party")
    expected_output = (
        "This is a test about animals.\n"
        "foo: bar\n"
        "baz: foo\n"
        "Now you try to talk about party."
    )
    assert output == expected_output


def test_partial() -> None:
    """Test prompt can be partialed."""
    prefix = "This is a test about {content}."
    suffix = "Now you try to talk about {new_content}."
    examples = [
        {"question": "foo", "answer": "bar"},
        {"question": "baz", "answer": "foo"},
    ]
    prompt = FewShotPromptTemplate(
        suffix=suffix,
        prefix=prefix,
        input_variables=["content", "new_content"],
        examples=examples,
        example_prompt=EXAMPLE_PROMPT,
        example_separator="\n",
    )
    new_prompt = prompt.partial(content="foo")
    new_output = new_prompt.format(new_content="party")
    expected_output = (
        "This is a test about foo.\n"
        "foo: bar\n"
        "baz: foo\n"
        "Now you try to talk about party."
    )
    assert new_output == expected_output
    output = prompt.format(new_content="party", content="bar")
    expected_output = (
        "This is a test about bar.\n"
        "foo: bar\n"
        "baz: foo\n"
        "Now you try to talk about party."
    )
    assert output == expected_output


@pytest.mark.requires("jinja2")
def test_prompt_jinja2_functionality(
    example_jinja2_prompt: tuple[PromptTemplate, list[dict[str, str]]],
) -> None:
    prefix = "Starting with {{ foo }}"
    suffix = "Ending with {{ bar }}"

    prompt = FewShotPromptTemplate(
        input_variables=["foo", "bar"],
        suffix=suffix,
        prefix=prefix,
        examples=example_jinja2_prompt[1],
        example_prompt=example_jinja2_prompt[0],
        template_format="jinja2",
    )
    output = prompt.format(foo="hello", bar="bye")
    expected_output = (
        "Starting with hello\n\nhappy: sad\n\ntall: short\n\nEnding with bye"
    )

    assert output == expected_output


@pytest.mark.requires("jinja2")
def test_prompt_jinja2_missing_input_variables(
    example_jinja2_prompt: tuple[PromptTemplate, list[dict[str, str]]],
) -> None:
    """Test error is raised when input variables are not provided."""
    prefix = "Starting with {{ foo }}"
    suffix = "Ending with {{ bar }}"

    # Test when missing in suffix
    with pytest.warns(UserWarning):
        FewShotPromptTemplate(
            input_variables=[],
            suffix=suffix,
            examples=example_jinja2_prompt[1],
            example_prompt=example_jinja2_prompt[0],
            template_format="jinja2",
            validate_template=True,
        )
    assert FewShotPromptTemplate(
        input_variables=[],
        suffix=suffix,
        examples=example_jinja2_prompt[1],
        example_prompt=example_jinja2_prompt[0],
        template_format="jinja2",
    ).input_variables == ["bar"]

    # Test when missing in prefix
    with pytest.warns(UserWarning):
        FewShotPromptTemplate(
            input_variables=["bar"],
            suffix=suffix,
            prefix=prefix,
            examples=example_jinja2_prompt[1],
            example_prompt=example_jinja2_prompt[0],
            template_format="jinja2",
            validate_template=True,
        )
    assert FewShotPromptTemplate(
        input_variables=["bar"],
        suffix=suffix,
        prefix=prefix,
        examples=example_jinja2_prompt[1],
        example_prompt=example_jinja2_prompt[0],
        template_format="jinja2",
    ).input_variables == ["bar", "foo"]


@pytest.mark.requires("jinja2")
def test_prompt_jinja2_extra_input_variables(
    example_jinja2_prompt: tuple[PromptTemplate, list[dict[str, str]]],
) -> None:
    """Test error is raised when there are too many input variables."""
    prefix = "Starting with {{ foo }}"
    suffix = "Ending with {{ bar }}"
    with pytest.warns(UserWarning):
        FewShotPromptTemplate(
            input_variables=["bar", "foo", "extra", "thing"],
            suffix=suffix,
            prefix=prefix,
            examples=example_jinja2_prompt[1],
            example_prompt=example_jinja2_prompt[0],
            template_format="jinja2",
            validate_template=True,
        )
    assert FewShotPromptTemplate(
        input_variables=["bar", "foo", "extra", "thing"],
        suffix=suffix,
        prefix=prefix,
        examples=example_jinja2_prompt[1],
        example_prompt=example_jinja2_prompt[0],
        template_format="jinja2",
    ).input_variables == ["bar", "foo"]


async def test_few_shot_chat_message_prompt_template() -> None:
    """Tests for few shot chat message template."""
    examples = [
        {"input": "2+2", "output": "4"},
        {"input": "2+3", "output": "5"},
    ]

    example_prompt = ChatPromptTemplate.from_messages(
        [
            HumanMessagePromptTemplate.from_template("{input}"),
            AIMessagePromptTemplate.from_template("{output}"),
        ]
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        input_variables=["input"],
        example_prompt=example_prompt,
        examples=examples,
    )
    final_prompt: ChatPromptTemplate = (
        SystemMessagePromptTemplate.from_template("You are a helpful AI Assistant")
        + few_shot_prompt
        + HumanMessagePromptTemplate.from_template("{input}")
    )

    expected = [
        SystemMessage(content="You are a helpful AI Assistant", additional_kwargs={}),
        HumanMessage(content="2+2", additional_kwargs={}, example=False),
        AIMessage(content="4", additional_kwargs={}, example=False),
        HumanMessage(content="2+3", additional_kwargs={}, example=False),
        AIMessage(content="5", additional_kwargs={}, example=False),
        HumanMessage(content="100 + 1", additional_kwargs={}, example=False),
    ]

    messages = final_prompt.format_messages(input="100 + 1")
    assert messages == expected
    messages = await final_prompt.aformat_messages(input="100 + 1")
    assert messages == expected


class AsIsSelector(BaseExampleSelector):
    """An example selector for testing purposes.

    This selector returns the examples as-is.
    """

    def __init__(self, examples: Sequence[dict[str, str]]) -> None:
        """Initializes the selector."""
        self.examples = examples

    def add_example(self, example: dict[str, str]) -> Any:
        raise NotImplementedError

    def select_examples(self, input_variables: dict[str, str]) -> list[dict]:
        return list(self.examples)


def test_few_shot_prompt_template_with_selector() -> None:
    """Tests for few shot chat message template with an example selector."""
    examples = [
        {"question": "foo", "answer": "bar"},
        {"question": "baz", "answer": "foo"},
    ]
    example_selector = AsIsSelector(examples)

    few_shot_prompt = FewShotPromptTemplate(
        input_variables=["foo"],
        suffix="This is a {foo} test.",
        example_prompt=EXAMPLE_PROMPT,
        example_selector=example_selector,
    )
    messages = few_shot_prompt.format(foo="bar")
    assert messages == "foo: bar\n\nbaz: foo\n\nThis is a bar test."


def test_few_shot_chat_message_prompt_template_with_selector() -> None:
    """Tests for few shot chat message template with an example selector."""
    examples = [
        {"input": "2+2", "output": "4"},
        {"input": "2+3", "output": "5"},
    ]
    example_selector = AsIsSelector(examples)
    example_prompt = ChatPromptTemplate.from_messages(
        [
            HumanMessagePromptTemplate.from_template("{input}"),
            AIMessagePromptTemplate.from_template("{output}"),
        ]
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        input_variables=["input"],
        example_prompt=example_prompt,
        example_selector=example_selector,
    )
    final_prompt: ChatPromptTemplate = (
        SystemMessagePromptTemplate.from_template("You are a helpful AI Assistant")
        + few_shot_prompt
        + HumanMessagePromptTemplate.from_template("{input}")
    )
    expected = [
        SystemMessage(content="You are a helpful AI Assistant", additional_kwargs={}),
        HumanMessage(content="2+2", additional_kwargs={}, example=False),
        AIMessage(content="4", additional_kwargs={}, example=False),
        HumanMessage(content="2+3", additional_kwargs={}, example=False),
        AIMessage(content="5", additional_kwargs={}, example=False),
        HumanMessage(content="100 + 1", additional_kwargs={}, example=False),
    ]
    messages = final_prompt.format_messages(input="100 + 1")
    assert messages == expected


def test_few_shot_chat_message_prompt_template_infer_input_variables() -> None:
    """Check that it can infer input variables if not provided."""
    examples = [
        {"input": "2+2", "output": "4"},
        {"input": "2+3", "output": "5"},
    ]
    example_selector = AsIsSelector(examples)
    example_prompt = ChatPromptTemplate.from_messages(
        [
            HumanMessagePromptTemplate.from_template("{input}"),
            AIMessagePromptTemplate.from_template("{output}"),
        ]
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        example_selector=example_selector,
    )

    # The prompt template does not have any inputs! They
    # have already been filled in.
    assert few_shot_prompt.input_variables == []


class AsyncAsIsSelector(BaseExampleSelector):
    """An example selector for testing purposes.

    This selector returns the examples as-is.
    """

    def __init__(self, examples: Sequence[dict[str, str]]) -> None:
        """Initializes the selector."""
        self.examples = examples

    def add_example(self, example: dict[str, str]) -> Any:
        raise NotImplementedError

    def select_examples(self, input_variables: dict[str, str]) -> list[dict]:
        raise NotImplementedError

    async def aselect_examples(self, input_variables: dict[str, str]) -> list[dict]:
        return list(self.examples)


async def test_few_shot_prompt_template_with_selector_async() -> None:
    """Tests for few shot chat message template with an example selector."""
    examples = [
        {"question": "foo", "answer": "bar"},
        {"question": "baz", "answer": "foo"},
    ]
    example_selector = AsyncAsIsSelector(examples)

    few_shot_prompt = FewShotPromptTemplate(
        input_variables=["foo"],
        suffix="This is a {foo} test.",
        example_prompt=EXAMPLE_PROMPT,
        example_selector=example_selector,
    )
    messages = await few_shot_prompt.aformat(foo="bar")
    assert messages == "foo: bar\n\nbaz: foo\n\nThis is a bar test."


async def test_few_shot_chat_message_prompt_template_with_selector_async() -> None:
    """Tests for few shot chat message template with an async example selector."""
    examples = [
        {"input": "2+2", "output": "4"},
        {"input": "2+3", "output": "5"},
    ]
    example_selector = AsyncAsIsSelector(examples)
    example_prompt = ChatPromptTemplate.from_messages(
        [
            HumanMessagePromptTemplate.from_template("{input}"),
            AIMessagePromptTemplate.from_template("{output}"),
        ]
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        input_variables=["input"],
        example_prompt=example_prompt,
        example_selector=example_selector,
    )
    final_prompt: ChatPromptTemplate = (
        SystemMessagePromptTemplate.from_template("You are a helpful AI Assistant")
        + few_shot_prompt
        + HumanMessagePromptTemplate.from_template("{input}")
    )
    expected = [
        SystemMessage(content="You are a helpful AI Assistant", additional_kwargs={}),
        HumanMessage(content="2+2", additional_kwargs={}, example=False),
        AIMessage(content="4", additional_kwargs={}, example=False),
        HumanMessage(content="2+3", additional_kwargs={}, example=False),
        AIMessage(content="5", additional_kwargs={}, example=False),
        HumanMessage(content="100 + 1", additional_kwargs={}, example=False),
    ]
    messages = await final_prompt.aformat_messages(input="100 + 1")
    assert messages == expected
