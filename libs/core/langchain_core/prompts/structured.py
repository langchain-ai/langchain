"""Structured prompt template for a language model."""

from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from typing import (
    Any,
)

from pydantic import BaseModel, Field
from typing_extensions import override

from langchain_core._api.beta_decorator import beta
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    MessageLikeRepresentation,
)
from langchain_core.prompts.string import PromptTemplateFormat
from langchain_core.runnables.base import (
    Other,
    Runnable,
    RunnableSequence,
    RunnableSerializable,
)
from langchain_core.utils import get_pydantic_field_names


@beta()
class StructuredPrompt(ChatPromptTemplate):
    """Structured prompt template for a language model."""

    schema_: dict | type
    """Schema for the structured prompt."""

    structured_output_kwargs: dict[str, Any] = Field(default_factory=dict)

    def __init__(
        self,
        messages: Sequence[MessageLikeRepresentation],
        schema_: dict | type[BaseModel] | None = None,
        *,
        structured_output_kwargs: dict[str, Any] | None = None,
        template_format: PromptTemplateFormat = "f-string",
        **kwargs: Any,
    ) -> None:
        """Create a structured prompt template.

        Args:
            messages: Sequence of messages.
            schema_: Schema for the structured prompt.
            structured_output_kwargs: Additional kwargs for structured output.
            template_format: Template format for the prompt.

        Raises:
            ValueError: If schema is not provided.
        """
        schema_ = schema_ or kwargs.pop("schema", None)
        if not schema_:
            err_msg = (
                "Must pass in a non-empty structured output schema. Received: "
                f"{schema_}"
            )
            raise ValueError(err_msg)
        structured_output_kwargs = structured_output_kwargs or {}
        for k in set(kwargs).difference(get_pydantic_field_names(self.__class__)):
            structured_output_kwargs[k] = kwargs.pop(k)
        super().__init__(
            messages=messages,
            schema_=schema_,
            structured_output_kwargs=structured_output_kwargs,
            template_format=template_format,
            **kwargs,
        )

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        For example, if the class is `langchain.llms.openai.OpenAI`, then the namespace
        is `["langchain", "llms", "openai"]`

        Returns:
            The namespace of the LangChain object.
        """
        return cls.__module__.split(".")

    @classmethod
    def from_messages_and_schema(
        cls,
        messages: Sequence[MessageLikeRepresentation],
        schema: dict | type,
        **kwargs: Any,
    ) -> ChatPromptTemplate:
        """Create a chat prompt template from a variety of message formats.

        Examples:
            Instantiation from a list of message templates:

            ```python
            from langchain_core.prompts import StructuredPrompt


            class OutputSchema(BaseModel):
                name: str
                value: int


            template = StructuredPrompt(
                [
                    ("human", "Hello, how are you?"),
                    ("ai", "I'm doing well, thanks!"),
                    ("human", "That's good to hear."),
                ],
                OutputSchema,
            )
            ```

        Args:
            messages: Sequence of message representations.

                A message can be represented using the following formats:

                1. `BaseMessagePromptTemplate`
                2. `BaseMessage`
                3. 2-tuple of `(message type, template)`; e.g.,
                    `("human", "{user_input}")`
                4. 2-tuple of `(message class, template)`
                5. A string which is shorthand for `("human", template)`; e.g.,
                    `"{user_input}"`
            schema: A dictionary representation of function call, or a Pydantic model.
            **kwargs: Any additional kwargs to pass through to
                `ChatModel.with_structured_output(schema, **kwargs)`.

        Returns:
            A structured prompt template
        """
        return cls(messages, schema, **kwargs)

    @override
    def __or__(
        self,
        other: Runnable[Any, Other]
        | Callable[[Iterator[Any]], Iterator[Other]]
        | Callable[[AsyncIterator[Any]], AsyncIterator[Other]]
        | Callable[[Any], Other]
        | Mapping[str, Runnable[Any, Other] | Callable[[Any], Other] | Any],
    ) -> RunnableSerializable[dict, Other]:
        return self.pipe(other)

    def pipe(
        self,
        *others: Runnable[Any, Other]
        | Callable[[Iterator[Any]], Iterator[Other]]
        | Callable[[AsyncIterator[Any]], AsyncIterator[Other]]
        | Callable[[Any], Other]
        | Mapping[str, Runnable[Any, Other] | Callable[[Any], Other] | Any],
        name: str | None = None,
    ) -> RunnableSerializable[dict, Other]:
        """Pipe the structured prompt to a language model.

        Args:
            others: The language model to pipe the structured prompt to.
            name: The name of the pipeline.

        Returns:
            A `RunnableSequence` object.

        Raises:
            NotImplementedError: If the first element of `others` is not a language
                model.
        """
        if (others and isinstance(others[0], BaseLanguageModel)) or hasattr(
            others[0], "with_structured_output"
        ):
            return RunnableSequence(
                self,
                others[0].with_structured_output(
                    self.schema_, **self.structured_output_kwargs
                ),
                *others[1:],
                name=name,
            )
        msg = "Structured prompts need to be piped to a language model."
        raise NotImplementedError(msg)
