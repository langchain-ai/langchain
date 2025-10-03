"""Prompt template that contains few shot examples."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)
from typing_extensions import override

from langchain_core.example_selectors import BaseExampleSelector
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.prompts.chat import BaseChatPromptTemplate
from langchain_core.prompts.message import BaseMessagePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.string import (
    DEFAULT_FORMATTER_MAPPING,
    StringPromptTemplate,
    check_valid_template,
    get_template_variables,
)

if TYPE_CHECKING:
    from pathlib import Path

    from typing_extensions import Self


class _FewShotPromptTemplateMixin(BaseModel):
    """Prompt template that contains few shot examples."""

    examples: Optional[list[dict]] = None
    """Examples to format into the prompt.
    Either this or example_selector should be provided."""

    example_selector: Optional[BaseExampleSelector] = None
    """ExampleSelector to choose the examples to format into the prompt.
    Either this or examples should be provided."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def check_examples_and_selector(cls, values: dict) -> Any:
        """Check that one and only one of examples/example_selector are provided.

        Args:
            values: The values to check.

        Returns:
            The values if they are valid.

        Raises:
            ValueError: If neither or both examples and example_selector are provided.
            ValueError: If both examples and example_selector are provided.
        """
        examples = values.get("examples")
        example_selector = values.get("example_selector")
        if examples and example_selector:
            msg = "Only one of 'examples' and 'example_selector' should be provided"
            raise ValueError(msg)

        if examples is None and example_selector is None:
            msg = "One of 'examples' and 'example_selector' should be provided"
            raise ValueError(msg)

        return values

    def _get_examples(self, **kwargs: Any) -> list[dict]:
        """Get the examples to use for formatting the prompt.

        Args:
            **kwargs: Keyword arguments to be passed to the example selector.

        Returns:
            List of examples.

        Raises:
            ValueError: If neither examples nor example_selector are provided.
        """
        if self.examples is not None:
            return self.examples
        if self.example_selector is not None:
            return self.example_selector.select_examples(kwargs)
        msg = "One of 'examples' and 'example_selector' should be provided"
        raise ValueError(msg)

    async def _aget_examples(self, **kwargs: Any) -> list[dict]:
        """Async get the examples to use for formatting the prompt.

        Args:
            **kwargs: Keyword arguments to be passed to the example selector.

        Returns:
            List of examples.

        Raises:
            ValueError: If neither examples nor example_selector are provided.
        """
        if self.examples is not None:
            return self.examples
        if self.example_selector is not None:
            return await self.example_selector.aselect_examples(kwargs)
        msg = "One of 'examples' and 'example_selector' should be provided"
        raise ValueError(msg)


class FewShotPromptTemplate(_FewShotPromptTemplateMixin, StringPromptTemplate):
    """Prompt template that contains few shot examples."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return False as this class is not serializable."""
        return False

    validate_template: bool = False
    """Whether or not to try validating the template."""

    example_prompt: PromptTemplate
    """PromptTemplate used to format an individual example."""

    suffix: str
    """A prompt template string to put after the examples."""

    example_separator: str = "\n\n"
    """String separator used to join the prefix, the examples, and suffix."""

    prefix: str = ""
    """A prompt template string to put before the examples."""

    template_format: Literal["f-string", "jinja2"] = "f-string"
    """The format of the prompt template. Options are: 'f-string', 'jinja2'."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the few shot prompt template."""
        if "input_variables" not in kwargs and "example_prompt" in kwargs:
            kwargs["input_variables"] = kwargs["example_prompt"].input_variables
        super().__init__(**kwargs)

    @model_validator(mode="after")
    def template_is_valid(self) -> Self:
        """Check that prefix, suffix, and input variables are consistent."""
        if self.validate_template:
            check_valid_template(
                self.prefix + self.suffix,
                self.template_format,
                self.input_variables + list(self.partial_variables),
            )
        elif self.template_format:
            self.input_variables = [
                var
                for var in get_template_variables(
                    self.prefix + self.suffix, self.template_format
                )
                if var not in self.partial_variables
            ]
        return self

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    def format(self, **kwargs: Any) -> str:
        """Format the prompt with inputs generating a string.

        Use this method to generate a string representation of a prompt.

        Args:
            **kwargs: keyword arguments to use for formatting.

        Returns:
            A string representation of the prompt.
        """
        kwargs = self._merge_partial_and_user_variables(**kwargs)
        # Get the examples to use.
        examples = self._get_examples(**kwargs)
        examples = [
            {k: e[k] for k in self.example_prompt.input_variables} for e in examples
        ]
        # Format the examples.
        example_strings = [
            self.example_prompt.format(**example) for example in examples
        ]
        # Create the overall template.
        pieces = [self.prefix, *example_strings, self.suffix]
        template = self.example_separator.join([piece for piece in pieces if piece])

        # Format the template with the input variables.
        return DEFAULT_FORMATTER_MAPPING[self.template_format](template, **kwargs)

    async def aformat(self, **kwargs: Any) -> str:
        """Async format the prompt with inputs generating a string.

        Use this method to generate a string representation of a prompt.

        Args:
            **kwargs: keyword arguments to use for formatting.

        Returns:
            A string representation of the prompt.
        """
        kwargs = self._merge_partial_and_user_variables(**kwargs)
        # Get the examples to use.
        examples = await self._aget_examples(**kwargs)
        examples = [
            {k: e[k] for k in self.example_prompt.input_variables} for e in examples
        ]
        # Format the examples.
        example_strings = [
            await self.example_prompt.aformat(**example) for example in examples
        ]
        # Create the overall template.
        pieces = [self.prefix, *example_strings, self.suffix]
        template = self.example_separator.join([piece for piece in pieces if piece])

        # Format the template with the input variables.
        return DEFAULT_FORMATTER_MAPPING[self.template_format](template, **kwargs)

    @property
    def _prompt_type(self) -> str:
        """Return the prompt type key."""
        return "few_shot"

    def save(self, file_path: Union[Path, str]) -> None:
        """Save the prompt template to a file.

        Args:
            file_path: The path to save the prompt template to.

        Raises:
            ValueError: If example_selector is provided.
        """
        if self.example_selector:
            msg = "Saving an example selector is not currently supported"
            raise ValueError(msg)
        return super().save(file_path)


class FewShotChatMessagePromptTemplate(
    BaseChatPromptTemplate, _FewShotPromptTemplateMixin
):
    """Chat prompt template that supports few-shot examples.

    The high level structure of produced by this prompt template is a list of messages
    consisting of prefix message(s), example message(s), and suffix message(s).

    This structure enables creating a conversation with intermediate examples like:

        System: You are a helpful AI Assistant
        Human: What is 2+2?
        AI: 4
        Human: What is 2+3?
        AI: 5
        Human: What is 4+4?

    This prompt template can be used to generate a fixed list of examples or else
    to dynamically select examples based on the input.

    Examples:
        Prompt template with a fixed list of examples (matching the sample
        conversation above):

        .. code-block:: python

            from langchain_core.prompts import (
                FewShotChatMessagePromptTemplate,
                ChatPromptTemplate,
            )

            examples = [
                {"input": "2+2", "output": "4"},
                {"input": "2+3", "output": "5"},
            ]

            example_prompt = ChatPromptTemplate.from_messages(
                [
                    ("human", "What is {input}?"),
                    ("ai", "{output}"),
                ]
            )

            few_shot_prompt = FewShotChatMessagePromptTemplate(
                examples=examples,
                # This is a prompt template used to format each individual example.
                example_prompt=example_prompt,
            )

            final_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful AI Assistant"),
                    few_shot_prompt,
                    ("human", "{input}"),
                ]
            )
            final_prompt.format(input="What is 4+4?")

        Prompt template with dynamically selected examples:

        .. code-block:: python

            from langchain_core.prompts import SemanticSimilarityExampleSelector
            from langchain_core.embeddings import OpenAIEmbeddings
            from langchain_core.vectorstores import Chroma

            examples = [
                {"input": "2+2", "output": "4"},
                {"input": "2+3", "output": "5"},
                {"input": "2+4", "output": "6"},
                # ...
            ]

            to_vectorize = [" ".join(example.values()) for example in examples]
            embeddings = OpenAIEmbeddings()
            vectorstore = Chroma.from_texts(
                to_vectorize, embeddings, metadatas=examples
            )
            example_selector = SemanticSimilarityExampleSelector(
                vectorstore=vectorstore
            )

            from langchain_core import SystemMessage
            from langchain_core.prompts import HumanMessagePromptTemplate
            from langchain_core.prompts.few_shot import FewShotChatMessagePromptTemplate

            few_shot_prompt = FewShotChatMessagePromptTemplate(
                # Which variable(s) will be passed to the example selector.
                input_variables=["input"],
                example_selector=example_selector,
                # Define how each example will be formatted.
                # In this case, each example will become 2 messages:
                # 1 human, and 1 AI
                example_prompt=(
                    HumanMessagePromptTemplate.from_template("{input}")
                    + AIMessagePromptTemplate.from_template("{output}")
                ),
            )
            # Define the overall prompt.
            final_prompt = (
                SystemMessagePromptTemplate.from_template(
                    "You are a helpful AI Assistant"
                )
                + few_shot_prompt
                + HumanMessagePromptTemplate.from_template("{input}")
            )
            # Show the prompt
            print(final_prompt.format_messages(input="What's 3+3?"))  # noqa: T201

            # Use within an LLM
            from langchain_core.chat_models import ChatAnthropic

            chain = final_prompt | ChatAnthropic(model="claude-3-haiku-20240307")
            chain.invoke({"input": "What's 3+3?"})

    """

    input_variables: list[str] = Field(default_factory=list)
    """A list of the names of the variables the prompt template will use
    to pass to the example_selector, if provided."""

    example_prompt: Union[BaseMessagePromptTemplate, BaseChatPromptTemplate]
    """The class to format each example."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return False as this class is not serializable."""
        return False

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """Format kwargs into a list of messages.

        Args:
            **kwargs: keyword arguments to use for filling in templates in messages.

        Returns:
            A list of formatted messages with all template variables filled in.
        """
        # Get the examples to use.
        examples = self._get_examples(**kwargs)
        examples = [
            {k: e[k] for k in self.example_prompt.input_variables} for e in examples
        ]
        # Format the examples.
        return [
            message
            for example in examples
            for message in self.example_prompt.format_messages(**example)
        ]

    async def aformat_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """Async format kwargs into a list of messages.

        Args:
            **kwargs: keyword arguments to use for filling in templates in messages.

        Returns:
            A list of formatted messages with all template variables filled in.
        """
        # Get the examples to use.
        examples = await self._aget_examples(**kwargs)
        examples = [
            {k: e[k] for k in self.example_prompt.input_variables} for e in examples
        ]
        # Format the examples.
        return [
            message
            for example in examples
            for message in await self.example_prompt.aformat_messages(**example)
        ]

    def format(self, **kwargs: Any) -> str:
        """Format the prompt with inputs generating a string.

        Use this method to generate a string representation of a prompt consisting
        of chat messages.

        Useful for feeding into a string-based completion language model or debugging.

        Args:
            **kwargs: keyword arguments to use for formatting.

        Returns:
            A string representation of the prompt
        """
        messages = self.format_messages(**kwargs)
        return get_buffer_string(messages)

    async def aformat(self, **kwargs: Any) -> str:
        """Async format the prompt with inputs generating a string.

        Use this method to generate a string representation of a prompt consisting
        of chat messages.

        Useful for feeding into a string-based completion language model or debugging.

        Args:
            **kwargs: keyword arguments to use for formatting.

        Returns:
            A string representation of the prompt
        """
        messages = await self.aformat_messages(**kwargs)
        return get_buffer_string(messages)

    @override
    def pretty_repr(self, html: bool = False) -> str:
        """Return a pretty representation of the prompt template.

        Args:
            html: Whether or not to return an HTML formatted string.

        Returns:
            A pretty representation of the prompt template.
        """
        raise NotImplementedError
