from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Type,
    TypeVar,
    Union,
)

from typing_extensions import TypeAlias

from langchain_core._api import deprecated
from langchain_core.messages import (
    AnyMessage,
    BaseMessage,
    MessageLikeRepresentation,
    get_buffer_string,
)
from langchain_core.prompt_values import PromptValue
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_core.runnables import Runnable, RunnableSerializable
from langchain_core.utils import get_pydantic_field_names

if TYPE_CHECKING:
    from langchain_core.caches import BaseCache
    from langchain_core.callbacks import Callbacks
    from langchain_core.outputs import LLMResult


@lru_cache(maxsize=None)  # Cache the tokenizer
def get_tokenizer() -> Any:
    try:
        from transformers import GPT2TokenizerFast  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "Could not import transformers python package. "
            "This is needed in order to calculate get_token_ids. "
            "Please install it with `pip install transformers`."
        )
    # create a GPT-2 tokenizer instance
    return GPT2TokenizerFast.from_pretrained("gpt2")


def _get_token_ids_default_method(text: str) -> List[int]:
    """Encode the text into token IDs."""
    # get the cached tokenizer
    tokenizer = get_tokenizer()

    # tokenize the text using the GPT-2 tokenizer
    return tokenizer.encode(text)


LanguageModelInput = Union[PromptValue, str, Sequence[MessageLikeRepresentation]]
LanguageModelOutput = Union[BaseMessage, str]
LanguageModelLike = Runnable[LanguageModelInput, LanguageModelOutput]
LanguageModelOutputVar = TypeVar("LanguageModelOutputVar", BaseMessage, str)


def _get_verbosity() -> bool:
    from langchain_core.globals import get_verbose

    return get_verbose()


class BaseLanguageModel(
    RunnableSerializable[LanguageModelInput, LanguageModelOutputVar], ABC
):
    """Abstract base class for interfacing with language models.

    All language model wrappers inherit from BaseLanguageModel.
    """

    cache: Union[BaseCache, bool, None] = None
    """Whether to cache the response.
    
    * If true, will use the global cache.
    * If false, will not use a cache
    * If None, will use the global cache if it's set, otherwise no cache.
    * If instance of BaseCache, will use the provided cache.
    
    Caching is not currently supported for streaming methods of models.
    """
    verbose: bool = Field(default_factory=_get_verbosity)
    """Whether to print out response text."""
    callbacks: Callbacks = Field(default=None, exclude=True)
    """Callbacks to add to the run trace."""
    tags: Optional[List[str]] = Field(default=None, exclude=True)
    """Tags to add to the run trace."""
    metadata: Optional[Dict[str, Any]] = Field(default=None, exclude=True)
    """Metadata to add to the run trace."""
    custom_get_token_ids: Optional[Callable[[str], List[int]]] = Field(
        default=None, exclude=True
    )
    """Optional encoder to use for counting tokens."""

    @validator("verbose", pre=True, always=True, allow_reuse=True)
    def set_verbose(cls, verbose: Optional[bool]) -> bool:
        """If verbose is None, set it.

        This allows users to pass in None as verbose to access the global setting.
        """
        if verbose is None:
            return _get_verbosity()
        else:
            return verbose

    @property
    def InputType(self) -> TypeAlias:
        """Get the input type for this runnable."""
        from langchain_core.prompt_values import (
            ChatPromptValueConcrete,
            StringPromptValue,
        )

        # This is a version of LanguageModelInput which replaces the abstract
        # base class BaseMessage with a union of its subclasses, which makes
        # for a much better schema.
        return Union[
            str,
            Union[StringPromptValue, ChatPromptValueConcrete],
            List[AnyMessage],
        ]

    @abstractmethod
    def generate_prompt(
        self,
        prompts: List[PromptValue],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Pass a sequence of prompts to the model and return model generations.

        This method should make use of batched calls for models that expose a batched
        API.

        Use this method when you want to:
            1. take advantage of batched calls,
            2. need more output from the model than just the top generated value,
            3. are building chains that are agnostic to the underlying language model
                type (e.g., pure text completion models vs chat models).

        Args:
            prompts: List of PromptValues. A PromptValue is an object that can be
                converted to match the format of any language model (string for pure
                text generation models and BaseMessages for chat models).
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of these substrings.
            callbacks: Callbacks to pass through. Used for executing additional
                functionality, such as logging or streaming, throughout generation.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            An LLMResult, which contains a list of candidate Generations for each input
                prompt and additional model provider-specific output.
        """

    @abstractmethod
    async def agenerate_prompt(
        self,
        prompts: List[PromptValue],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Asynchronously pass a sequence of prompts and return model generations.

        This method should make use of batched calls for models that expose a batched
        API.

        Use this method when you want to:
            1. take advantage of batched calls,
            2. need more output from the model than just the top generated value,
            3. are building chains that are agnostic to the underlying language model
                type (e.g., pure text completion models vs chat models).

        Args:
            prompts: List of PromptValues. A PromptValue is an object that can be
                converted to match the format of any language model (string for pure
                text generation models and BaseMessages for chat models).
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of these substrings.
            callbacks: Callbacks to pass through. Used for executing additional
                functionality, such as logging or streaming, throughout generation.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            An LLMResult, which contains a list of candidate Generations for each input
                prompt and additional model provider-specific output.
        """

    def with_structured_output(
        self, schema: Union[Dict, Type[BaseModel]], **kwargs: Any
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        """Not implemented on this class."""
        # Implement this on child class if there is a way of steering the model to
        # generate responses that match a given schema.
        raise NotImplementedError()

    @deprecated("0.1.7", alternative="invoke", removal="0.3.0")
    @abstractmethod
    def predict(
        self, text: str, *, stop: Optional[Sequence[str]] = None, **kwargs: Any
    ) -> str:
        """Pass a single string input to the model and return a string.

         Use this method when passing in raw text. If you want to pass in specific
            types of chat messages, use predict_messages.

        Args:
            text: String input to pass to the model.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of these substrings.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            Top model prediction as a string.
        """

    @deprecated("0.1.7", alternative="invoke", removal="0.3.0")
    @abstractmethod
    def predict_messages(
        self,
        messages: List[BaseMessage],
        *,
        stop: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        """Pass a message sequence to the model and return a message.

        Use this method when passing in chat messages. If you want to pass in raw text,
            use predict.

        Args:
            messages: A sequence of chat messages corresponding to a single model input.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of these substrings.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            Top model prediction as a message.
        """

    @deprecated("0.1.7", alternative="ainvoke", removal="0.3.0")
    @abstractmethod
    async def apredict(
        self, text: str, *, stop: Optional[Sequence[str]] = None, **kwargs: Any
    ) -> str:
        """Asynchronously pass a string to the model and return a string.

        Use this method when calling pure text generation models and only the top
            candidate generation is needed.

        Args:
            text: String input to pass to the model.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of these substrings.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            Top model prediction as a string.
        """

    @deprecated("0.1.7", alternative="ainvoke", removal="0.3.0")
    @abstractmethod
    async def apredict_messages(
        self,
        messages: List[BaseMessage],
        *,
        stop: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        """Asynchronously pass messages to the model and return a message.

        Use this method when calling chat models and only the top
            candidate generation is needed.

        Args:
            messages: A sequence of chat messages corresponding to a single model input.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of these substrings.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            Top model prediction as a message.
        """

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return self.lc_attributes

    def get_token_ids(self, text: str) -> List[int]:
        """Return the ordered ids of the tokens in a text.

        Args:
            text: The string input to tokenize.

        Returns:
            A list of ids corresponding to the tokens in the text, in order they occur
                in the text.
        """
        if self.custom_get_token_ids is not None:
            return self.custom_get_token_ids(text)
        else:
            return _get_token_ids_default_method(text)

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens present in the text.

        Useful for checking if an input will fit in a model's context window.

        Args:
            text: The string input to tokenize.

        Returns:
            The integer number of tokens in the text.
        """
        return len(self.get_token_ids(text))

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        """Get the number of tokens in the messages.

        Useful for checking if an input will fit in a model's context window.

        Args:
            messages: The message inputs to tokenize.

        Returns:
            The sum of the number of tokens across the messages.
        """
        return sum([self.get_num_tokens(get_buffer_string([m])) for m in messages])

    @classmethod
    def _all_required_field_names(cls) -> Set:
        """DEPRECATED: Kept for backwards compatibility.

        Use get_pydantic_field_names.
        """
        return get_pydantic_field_names(cls)
