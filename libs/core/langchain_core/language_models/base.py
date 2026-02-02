"""Base language models class."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from functools import cache
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeAlias,
    TypeVar,
    cast,
)

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import TypedDict, override

from langchain_core.caches import BaseCache  # noqa: TC001
from langchain_core.callbacks import Callbacks  # noqa: TC001
from langchain_core.globals import get_verbose
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    MessageLikeRepresentation,
    get_buffer_string,
)
from langchain_core.prompt_values import (
    ChatPromptValueConcrete,
    PromptValue,
    StringPromptValue,
)
from langchain_core.runnables import Runnable, RunnableSerializable

if TYPE_CHECKING:
    from langchain_core.outputs import LLMResult

try:
    from transformers import GPT2TokenizerFast  # type: ignore[import-not-found]

    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


class LangSmithParams(TypedDict, total=False):
    """LangSmith parameters for tracing."""

    ls_provider: str
    """Provider of the model."""

    ls_model_name: str
    """Name of the model."""

    ls_model_type: Literal["chat", "llm"]
    """Type of the model.

    Should be `'chat'` or `'llm'`.
    """

    ls_temperature: float | None
    """Temperature for generation."""

    ls_max_tokens: int | None
    """Max tokens for generation."""

    ls_stop: list[str] | None
    """Stop words for generation."""


@cache  # Cache the tokenizer
def get_tokenizer() -> Any:
    """Get a GPT-2 tokenizer instance.

    This function is cached to avoid re-loading the tokenizer every time it is called.

    Raises:
        ImportError: If the transformers package is not installed.

    Returns:
        The GPT-2 tokenizer instance.

    """
    if not _HAS_TRANSFORMERS:
        msg = (
            "Could not import transformers python package. "
            "This is needed in order to calculate get_token_ids. "
            "Please install it with `pip install transformers`."
        )
        raise ImportError(msg)
    # create a GPT-2 tokenizer instance
    return GPT2TokenizerFast.from_pretrained("gpt2")


_GPT2_TOKENIZER_WARNED = False


def _get_token_ids_default_method(text: str) -> list[int]:
    """Encode the text into token IDs using the fallback GPT-2 tokenizer."""
    global _GPT2_TOKENIZER_WARNED  # noqa: PLW0603
    if not _GPT2_TOKENIZER_WARNED:
        warnings.warn(
            "Using fallback GPT-2 tokenizer for token counting. "
            "Token counts may be inaccurate for non-GPT-2 models. "
            "For accurate counts, use a model-specific method if available.",
            stacklevel=3,
        )
        _GPT2_TOKENIZER_WARNED = True

    tokenizer = get_tokenizer()

    # Pass verbose=False to suppress the "Token indices sequence length is longer than
    # the specified maximum sequence length" warning from HuggingFace. This warning is
    # about GPT-2's 1024 token context limit, but we're only using the tokenizer for
    # counting, not for model input.
    return cast("list[int]", tokenizer.encode(text, verbose=False))


LanguageModelInput = PromptValue | str | Sequence[MessageLikeRepresentation]
"""Input to a language model."""

LanguageModelOutput = BaseMessage | str
"""Output from a language model."""

LanguageModelLike = Runnable[LanguageModelInput, LanguageModelOutput]
"""Input/output interface for a language model."""

LanguageModelOutputVar = TypeVar("LanguageModelOutputVar", AIMessage, str)
"""Type variable for the output of a language model."""


def _get_verbosity() -> bool:
    return get_verbose()


class BaseLanguageModel(
    RunnableSerializable[LanguageModelInput, LanguageModelOutputVar], ABC
):
    """Abstract base class for interfacing with language models.

    All language model wrappers inherited from `BaseLanguageModel`.

    """

    cache: BaseCache | bool | None = Field(default=None, exclude=True)
    """Whether to cache the response.

    * If `True`, will use the global cache.
    * If `False`, will not use a cache
    * If `None`, will use the global cache if it's set, otherwise no cache.
    * If instance of `BaseCache`, will use the provided cache.

    Caching is not currently supported for streaming methods of models.
    """

    verbose: bool = Field(default_factory=_get_verbosity, exclude=True, repr=False)
    """Whether to print out response text."""

    callbacks: Callbacks = Field(default=None, exclude=True)
    """Callbacks to add to the run trace."""

    tags: list[str] | None = Field(default=None, exclude=True)
    """Tags to add to the run trace."""

    metadata: dict[str, Any] | None = Field(default=None, exclude=True)
    """Metadata to add to the run trace."""

    custom_get_token_ids: Callable[[str], list[int]] | None = Field(
        default=None, exclude=True
    )
    """Optional encoder to use for counting tokens."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @field_validator("verbose", mode="before")
    def set_verbose(cls, verbose: bool | None) -> bool:  # noqa: FBT001
        """If verbose is `None`, set it.

        This allows users to pass in `None` as verbose to access the global setting.

        Args:
            verbose: The verbosity setting to use.

        Returns:
            The verbosity setting to use.

        """
        if verbose is None:
            return _get_verbosity()
        return verbose

    @property
    @override
    def InputType(self) -> TypeAlias:
        """Get the input type for this `Runnable`."""
        # This is a version of LanguageModelInput which replaces the abstract
        # base class BaseMessage with a union of its subclasses, which makes
        # for a much better schema.
        return str | StringPromptValue | ChatPromptValueConcrete | list[AnyMessage]

    @abstractmethod
    def generate_prompt(
        self,
        prompts: list[PromptValue],
        stop: list[str] | None = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Pass a sequence of prompts to the model and return model generations.

        This method should make use of batched calls for models that expose a batched
        API.

        Use this method when you want to:

        1. Take advantage of batched calls,
        2. Need more output from the model than just the top generated value,
        3. Are building chains that are agnostic to the underlying language model
            type (e.g., pure text completion models vs chat models).

        Args:
            prompts: List of `PromptValue` objects.

                A `PromptValue` is an object that can be converted to match the format
                of any language model (string for pure text generation models and
                `BaseMessage` objects for chat models).
            stop: Stop words to use when generating.

                Model output is cut off at the first occurrence of any of these
                substrings.
            callbacks: `Callbacks` to pass through.

                Used for executing additional functionality, such as logging or
                streaming, throughout generation.
            **kwargs: Arbitrary additional keyword arguments.

                These are usually passed to the model provider API call.

        Returns:
            An `LLMResult`, which contains a list of candidate `Generation` objects for
                each input prompt and additional model provider-specific output.

        """

    @abstractmethod
    async def agenerate_prompt(
        self,
        prompts: list[PromptValue],
        stop: list[str] | None = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Asynchronously pass a sequence of prompts and return model generations.

        This method should make use of batched calls for models that expose a batched
        API.

        Use this method when you want to:

        1. Take advantage of batched calls,
        2. Need more output from the model than just the top generated value,
        3. Are building chains that are agnostic to the underlying language model
            type (e.g., pure text completion models vs chat models).

        Args:
            prompts: List of `PromptValue` objects.

                A `PromptValue` is an object that can be converted to match the format
                of any language model (string for pure text generation models and
                `BaseMessage` objects for chat models).
            stop: Stop words to use when generating.

                Model output is cut off at the first occurrence of any of these
                substrings.
            callbacks: `Callbacks` to pass through.

                Used for executing additional functionality, such as logging or
                streaming, throughout generation.
            **kwargs: Arbitrary additional keyword arguments.

                These are usually passed to the model provider API call.

        Returns:
            An `LLMResult`, which contains a list of candidate `Generation` objects for
                each input prompt and additional model provider-specific output.

        """

    def with_structured_output(
        self, schema: dict | type, **kwargs: Any
    ) -> Runnable[LanguageModelInput, dict | BaseModel]:
        """Not implemented on this class."""
        # Implement this on child class if there is a way of steering the model to
        # generate responses that match a given schema.
        raise NotImplementedError

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return self.lc_attributes

    def get_token_ids(self, text: str) -> list[int]:
        """Return the ordered IDs of the tokens in a text.

        Args:
            text: The string input to tokenize.

        Returns:
            A list of IDs corresponding to the tokens in the text, in order they occur
                in the text.
        """
        if self.custom_get_token_ids is not None:
            return self.custom_get_token_ids(text)
        return _get_token_ids_default_method(text)

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens present in the text.

        Useful for checking if an input fits in a model's context window.

        This should be overridden by model-specific implementations to provide accurate
        token counts via model-specific tokenizers.

        Args:
            text: The string input to tokenize.

        Returns:
            The integer number of tokens in the text.

        """
        return len(self.get_token_ids(text))

    def get_num_tokens_from_messages(
        self,
        messages: list[BaseMessage],
        tools: Sequence | None = None,
    ) -> int:
        """Get the number of tokens in the messages.

        Useful for checking if an input fits in a model's context window.

        This should be overridden by model-specific implementations to provide accurate
        token counts via model-specific tokenizers.

        !!! note

            * The base implementation of `get_num_tokens_from_messages` ignores tool
                schemas.
            * The base implementation of `get_num_tokens_from_messages` adds additional
                prefixes to messages in represent user roles, which will add to the
                overall token count. Model-specific implementations may choose to
                handle this differently.

        Args:
            messages: The message inputs to tokenize.
            tools: If provided, sequence of dict, `BaseModel`, function, or
                `BaseTool` objects to be converted to tool schemas.

        Returns:
            The sum of the number of tokens across the messages.

        """
        if tools is not None:
            warnings.warn(
                "Counting tokens in tool schemas is not yet supported. Ignoring tools.",
                stacklevel=2,
            )
        return sum(self.get_num_tokens(get_buffer_string([m])) for m in messages)
