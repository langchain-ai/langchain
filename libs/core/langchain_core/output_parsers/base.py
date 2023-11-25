from __future__ import annotations

import asyncio
import functools
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
)

from typing_extensions import get_args

from langchain_core.messages import AnyMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.runnables import RunnableConfig, RunnableSerializable

if TYPE_CHECKING:
    from langchain_core.prompt_values import PromptValue

T = TypeVar("T")


class BaseLLMOutputParser(Generic[T], ABC):
    """Abstract base class for parsing the outputs of a model."""

    @abstractmethod
    def parse_result(self, result: List[Generation], *, partial: bool = False) -> T:
        """Parse a list of candidate model Generations into a specific format.

        Args:
            result: A list of Generations to be parsed. The Generations are assumed
                to be different candidate outputs for a single model input.

        Returns:
            Structured output.
        """

    async def aparse_result(
        self, result: List[Generation], *, partial: bool = False
    ) -> T:
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
    BaseLLMOutputParser, RunnableSerializable[Union[str, BaseMessage], T]
):
    """Base class to parse the output of an LLM call."""

    @property
    def InputType(self) -> Any:
        return Union[str, AnyMessage]

    @property
    def OutputType(self) -> Type[T]:
        # even though mypy complains this isn't valid,
        # it is good enough for pydantic to build the schema from
        return T  # type: ignore[misc]

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
        config: Optional[RunnableConfig] = None,
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


class BaseOutputParser(
    BaseLLMOutputParser, RunnableSerializable[Union[str, BaseMessage], T]
):
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

    @property
    def InputType(self) -> Any:
        return Union[str, AnyMessage]

    @property
    def OutputType(self) -> Type[T]:
        for cls in self.__class__.__orig_bases__:  # type: ignore[attr-defined]
            type_args = get_args(cls)
            if type_args and len(type_args) == 1:
                return type_args[0]

        raise TypeError(
            f"Runnable {self.__class__.__name__} doesn't have an inferable OutputType. "
            "Override the OutputType property to specify the output type."
        )

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
        config: Optional[RunnableConfig] = None,
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

    def parse_result(self, result: List[Generation], *, partial: bool = False) -> T:
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

    async def aparse_result(
        self, result: List[Generation], *, partial: bool = False
    ) -> T:
        """Parse a list of candidate model Generations into a specific format.

        The return value is parsed from only the first Generation in the result, which
            is assumed to be the highest-likelihood Generation.

        Args:
            result: A list of Generations to be parsed. The Generations are assumed
                to be different candidate outputs for a single model input.

        Returns:
            Structured output.
        """
        return await asyncio.get_running_loop().run_in_executor(
            None, functools.partial(self.parse_result, partial=partial), result
        )

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
        try:
            output_parser_dict["_type"] = self._type
        except NotImplementedError:
            pass
        return output_parser_dict
