from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    cast,
)
from typing_extensions import Unpack

from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.load.serializable import Serializable
from langchain.schema.messages import BaseMessage, get_buffer_string
from langchain.schema.output import LLMResult
from langchain.schema.prompt import PromptValue
from langchain.utils import get_pydantic_field_names
from langchain.schema.runnable import Runnable, RunnableConfig

from langchain.callbacks.streaming_iter import IteratorCallbackHandler

if TYPE_CHECKING:
    from langchain.callbacks.manager import Callbacks


def _get_token_ids_default_method(text: str) -> List[int]:
    """Encode the text into token IDs."""
    # TODO: this method may not be exact.
    # TODO: this method may differ based on model (eg codex).
    try:
        from transformers import GPT2TokenizerFast
    except ImportError:
        raise ImportError(
            "Could not import transformers python package. "
            "This is needed in order to calculate get_token_ids. "
            "Please install it with `pip install transformers`."
        )
    # create a GPT-2 tokenizer instance
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # tokenize the text using the GPT-2 tokenizer
    return tokenizer.encode(text)


class BaseLanguageModel(Serializable, Runnable[PromptValue, str], ABC):
    """Abstract base class for interfacing with language models.

    All language model wrappers inherit from BaseLanguageModel.

    Exposes three main methods:
    - generate_prompt: generate language model outputs for a sequence of prompt
        values. A prompt value is a model input that can be converted to any language
        model input format (string or messages).
    - predict: pass in a single string to a language model and return a string
        prediction.
    - predict_messages: pass in a sequence of BaseMessages (corresponding to a single
        model call) to a language model and return a BaseMessage prediction.

    Each of these has an equivalent asynchronous method.
    """

    def invoke(
        self,
        input: PromptValue,
        stop: Optional[List[str]] = None,
        **kwargs: Unpack[RunnableConfig],
    ) -> str:
        return self.generate_prompt([input], stop=stop, **kwargs).generations[0][0].text

    async def ainvoke(
        self,
        input: PromptValue,
        stop: Optional[List[str]] = None,
        **kwargs: Unpack[RunnableConfig],
    ) -> str:
        llm_result = await self.agenerate_prompt([input], stop=stop, **kwargs)
        return llm_result.generations[0][0].text

    def batch(
        self,
        inputs: List[PromptValue],
        config: Optional[RunnableConfig | List[RunnableConfig]] = None,
        max_concurrency: Optional[int] = None,
    ) -> List[str]:
        if isinstance(config, list):
            config = config[0]
        if config is None:
            config = {}

        llm_result = self.generate_prompt(inputs, **config)
        return [g[0].text for g in llm_result.generations]

    async def abatch(
        self,
        inputs: List[PromptValue],
        config: Optional[RunnableConfig | List[RunnableConfig]] = None,
        max_concurrency: Optional[int] = None,
    ) -> List[str]:
        if isinstance(config, list):
            config = config[0]
        if config is None:
            config = {}

        llm_result = await self.agenerate_prompt(inputs, **config)
        return [g[0].text for g in llm_result.generations]

    def stream(
        self,
        input: PromptValue,
        stop: Optional[List[str]] = None,
        **kwargs: Unpack[RunnableConfig],
    ) -> Iterator[str]:
        if not hasattr(self, "streaming"):
            # model doesn't support streaming, so use default implementation
            yield self.invoke(input, stop=stop, **kwargs)
        else:
            # enable streaming, if it's not already enabled
            original_streaming = cast(bool, self.streaming)  # type: ignore
            self.streaming = True

            # add iter callback handler to kwargs
            callbacks: Optional[Callbacks] = kwargs.pop("callbacks", None)
            callback_handler = IteratorCallbackHandler()
            if callbacks is None:
                kwargs["callbacks"] = [callback_handler]
            elif isinstance(callbacks, list):
                callbacks.append(callback_handler)
            else:
                callbacks.add_handler(callback_handler)

            with ThreadPoolExecutor(max_workers=1) as executor:
                # run the model non-blocking
                task = executor.submit(self.invoke, input, stop=stop, **kwargs)

                # yield tokens from the callback handler
                for token in callback_handler.iter():
                    yield token

                # block until the model is finished
                task.result()

            # disable streaming
            self.streaming = original_streaming

    async def astream(
        self,
        input: PromptValue,
        stop: Optional[List[str]] = None,
        **kwargs: Unpack[RunnableConfig],
    ) -> AsyncIterator[str]:
        if not hasattr(self, "streaming"):
            # model doesn't support streaming, so use default implementation
            yield await self.ainvoke(input, stop=stop, **kwargs)
        else:
            # enable streaming, if it's not already enabled
            original_streaming = cast(bool, self.streaming)  # type: ignore
            self.streaming = True

            # add aiter callback handler to kwargs
            callbacks: Optional[Callbacks] = kwargs.pop("callbacks", None)
            callback_handler = AsyncIteratorCallbackHandler()
            if callbacks is None:
                kwargs["callbacks"] = [callback_handler]
            elif isinstance(callbacks, list):
                callbacks.append(callback_handler)
            else:
                callbacks.add_handler(callback_handler)

            # run the model asynchronously
            task = asyncio.create_task(self.ainvoke(input, stop=stop, **kwargs))

            # yield tokens from the callback handler
            async for token in callback_handler.aiter():
                yield token

            # wait for the model to finish
            await task

            # restore original streaming value
            self.streaming = original_streaming

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

    @abstractmethod
    def predict(
        self, text: str, *, stop: Optional[Sequence[str]] = None, **kwargs: Any
    ) -> str:
        """Pass a single string input to the model and return a string prediction.

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

    @abstractmethod
    def predict_messages(
        self,
        messages: List[BaseMessage],
        *,
        stop: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        """Pass a message sequence to the model and return a message prediction.

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

    @abstractmethod
    async def apredict(
        self, text: str, *, stop: Optional[Sequence[str]] = None, **kwargs: Any
    ) -> str:
        """Asynchronously pass a string to the model and return a string prediction.

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

    @abstractmethod
    async def apredict_messages(
        self,
        messages: List[BaseMessage],
        *,
        stop: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        """Asynchronously pass messages to the model and return a message prediction.

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

    def get_token_ids(self, text: str) -> List[int]:
        """Return the ordered ids of the tokens in a text.

        Args:
            text: The string input to tokenize.

        Returns:
            A list of ids corresponding to the tokens in the text, in order they occur
                in the text.
        """
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
