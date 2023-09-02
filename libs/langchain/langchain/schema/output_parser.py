from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    TypeVar,
    Union,
)

from langchain.load.serializable import Serializable
from langchain.schema.messages import BaseMessage
from langchain.schema.output import ChatGeneration, Generation
from langchain.schema.prompt import PromptValue
from langchain.schema.runnable import Runnable, RunnableConfig

T = TypeVar("T")


class BaseLLMOutputParser(Serializable, Generic[T], ABC):
    """Abstract base class for parsing the outputs of a model."""

    @abstractmethod
    def parse_result(self, result: List[Generation]) -> T:
        """Parse a list of candidate model Generations into a specific format.

        Args:
            result: A list of Generations to be parsed. The Generations are assumed
                to be different candidate outputs for a single model input.

        Returns:
            Structured output.
        """

    async def aparse_result(self, result: List[Generation]) -> T:
        """Parse a list of candidate model Generations into a specific format.

        Args:
            result: A list of Generations to be parsed. The Generations are assumed
                to be different candidate outputs for a single model input.

        Returns:
            Structured output.
        """
        return await asyncio.get_running_loop().run_in_executor(
            None, self.parse_result, result
        )


class BaseGenerationOutputParser(
    BaseLLMOutputParser, Runnable[Union[str, BaseMessage], T]
):
    """Base class to parse the output of an LLM call."""

    def invoke(
        self, input: Union[str, BaseMessage], config: Optional[RunnableConfig] = None
    ) -> T:
        if isinstance(input, BaseMessage):
            return self._call_with_config(
                lambda inner_input: self.parse_result(
                    [ChatGeneration(message=inner_input)]
                ),
                input,
                config,
                run_type="parser",
            )
        else:
            return self._call_with_config(
                lambda inner_input: self.parse_result([Generation(text=inner_input)]),
                input,
                config,
                run_type="parser",
            )

    async def ainvoke(
        self,
        input: str | BaseMessage,
        config: RunnableConfig | None = None,
        **kwargs: Optional[Any],
    ) -> T:
        if isinstance(input, BaseMessage):
            return await self._acall_with_config(
                lambda inner_input: self.aparse_result(
                    [ChatGeneration(message=inner_input)]
                ),
                input,
                config,
                run_type="parser",
            )
        else:
            return await self._acall_with_config(
                lambda inner_input: self.aparse_result([Generation(text=inner_input)]),
                input,
                config,
                run_type="parser",
            )


class BaseOutputParser(BaseLLMOutputParser, Runnable[Union[str, BaseMessage], T]):
    """Base class to parse the output of an LLM call.

    Output parsers help structure language model responses.

    Example:
        .. code-block:: python

            class BooleanOutputParser(BaseOutputParser[bool]):
                true_val: str = "YES"
                false_val: str = "NO"

                def parse(self, text: str) -> bool:
                    cleaned_text = text.strip().upper()
                    if cleaned_text not in (self.true_val.upper(), self.false_val.upper()):
                        raise OutputParserException(
                            f"BooleanOutputParser expected output value to either be "
                            f"{self.true_val} or {self.false_val} (case-insensitive). "
                            f"Received {cleaned_text}."
                        )
                    return cleaned_text == self.true_val.upper()

                    @property
                    def _type(self) -> str:
                            return "boolean_output_parser"
    """  # noqa: E501

    def invoke(
        self, input: Union[str, BaseMessage], config: Optional[RunnableConfig] = None
    ) -> T:
        if isinstance(input, BaseMessage):
            return self._call_with_config(
                lambda inner_input: self.parse_result(
                    [ChatGeneration(message=inner_input)]
                ),
                input,
                config,
                run_type="parser",
            )
        else:
            return self._call_with_config(
                lambda inner_input: self.parse_result([Generation(text=inner_input)]),
                input,
                config,
                run_type="parser",
            )

    async def ainvoke(
        self,
        input: str | BaseMessage,
        config: RunnableConfig | None = None,
        **kwargs: Optional[Any],
    ) -> T:
        if isinstance(input, BaseMessage):
            return await self._acall_with_config(
                lambda inner_input: self.aparse_result(
                    [ChatGeneration(message=inner_input)]
                ),
                input,
                config,
                run_type="parser",
            )
        else:
            return await self._acall_with_config(
                lambda inner_input: self.aparse_result([Generation(text=inner_input)]),
                input,
                config,
                run_type="parser",
            )

    def parse_result(self, result: List[Generation]) -> T:
        """Parse a list of candidate model Generations into a specific format.

        The return value is parsed from only the first Generation in the result, which
            is assumed to be the highest-likelihood Generation.

        Args:
            result: A list of Generations to be parsed. The Generations are assumed
                to be different candidate outputs for a single model input.

        Returns:
            Structured output.
        """
        return self.parse(result[0].text)

    @abstractmethod
    def parse(self, text: str) -> T:
        """Parse a single string model output into some structure.

        Args:
            text: String output of a language model.

        Returns:
            Structured output.
        """

    async def aparse_result(self, result: List[Generation]) -> T:
        """Parse a list of candidate model Generations into a specific format.

        The return value is parsed from only the first Generation in the result, which
            is assumed to be the highest-likelihood Generation.

        Args:
            result: A list of Generations to be parsed. The Generations are assumed
                to be different candidate outputs for a single model input.

        Returns:
            Structured output.
        """
        return await self.aparse(result[0].text)

    async def aparse(self, text: str) -> T:
        """Parse a single string model output into some structure.

        Args:
            text: String output of a language model.

        Returns:
            Structured output.
        """
        return await asyncio.get_running_loop().run_in_executor(None, self.parse, text)

    # TODO: rename 'completion' -> 'text'.
    def parse_with_prompt(self, completion: str, prompt: PromptValue) -> Any:
        """Parse the output of an LLM call with the input prompt for context.

        The prompt is largely provided in the event the OutputParser wants
        to retry or fix the output in some way, and needs information from
        the prompt to do so.

        Args:
            completion: String output of a language model.
            prompt: Input PromptValue.

        Returns:
            Structured output
        """
        return self.parse(completion)

    def get_format_instructions(self) -> str:
        """Instructions on how the LLM output should be formatted."""
        raise NotImplementedError

    @property
    def _type(self) -> str:
        """Return the output parser type for serialization."""
        raise NotImplementedError(
            f"_type property is not implemented in class {self.__class__.__name__}."
            " This is required for serialization."
        )

    def dict(self, **kwargs: Any) -> Dict:
        """Return dictionary representation of output parser."""
        output_parser_dict = super().dict(**kwargs)
        output_parser_dict["_type"] = self._type
        return output_parser_dict


class BaseTransformOutputParser(BaseOutputParser[T]):
    """Base class for an output parser that can handle streaming input."""

    def _transform(self, input: Iterator[Union[str, BaseMessage]]) -> Iterator[T]:
        for chunk in input:
            if isinstance(chunk, BaseMessage):
                yield self.parse_result([ChatGeneration(message=chunk)])
            else:
                yield self.parse_result([Generation(text=chunk)])

    async def _atransform(
        self, input: AsyncIterator[Union[str, BaseMessage]]
    ) -> AsyncIterator[T]:
        async for chunk in input:
            if isinstance(chunk, BaseMessage):
                yield self.parse_result([ChatGeneration(message=chunk)])
            else:
                yield self.parse_result([Generation(text=chunk)])

    def transform(
        self,
        input: Iterator[Union[str, BaseMessage]],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Iterator[T]:
        yield from self._transform_stream_with_config(
            input, self._transform, config, run_type="parser"
        )

    async def atransform(
        self,
        input: AsyncIterator[Union[str, BaseMessage]],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AsyncIterator[T]:
        async for chunk in self._atransform_stream_with_config(
            input, self._atransform, config, run_type="parser"
        ):
            yield chunk


class StrOutputParser(BaseTransformOutputParser[str]):
    """OutputParser that parses LLMResult into the top likely string."""

    @property
    def lc_serializable(self) -> bool:
        """Whether the class LangChain serializable."""
        return True

    @property
    def _type(self) -> str:
        """Return the output parser type for serialization."""
        return "default"

    def parse(self, text: str) -> str:
        """Returns the input text with no changes."""
        return text


# TODO: Deprecate
NoOpOutputParser = StrOutputParser


class OutputParserException(ValueError):
    """Exception that output parsers should raise to signify a parsing error.

    This exists to differentiate parsing errors from other code or execution errors
    that also may arise inside the output parser. OutputParserExceptions will be
    available to catch and handle in ways to fix the parsing error, while other
    errors will be raised.

    Args:
        error: The error that's being re-raised or an error message.
        observation: String explanation of error which can be passed to a
            model to try and remediate the issue.
        llm_output: String model output which is error-ing.
        send_to_llm: Whether to send the observation and llm_output back to an Agent
            after an OutputParserException has been raised. This gives the underlying
            model driving the agent the context that the previous output was improperly
            structured, in the hopes that it will update the output to the correct
            format.
    """

    def __init__(
        self,
        error: Any,
        observation: Optional[str] = None,
        llm_output: Optional[str] = None,
        send_to_llm: bool = False,
    ):
        super(OutputParserException, self).__init__(error)
        if send_to_llm:
            if observation is None or llm_output is None:
                raise ValueError(
                    "Arguments 'observation' & 'llm_output'"
                    " are required if 'send_to_llm' is True"
                )
        self.observation = observation
        self.llm_output = llm_output
        self.send_to_llm = send_to_llm
