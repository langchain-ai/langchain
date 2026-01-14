"""Output parser that wraps another parser and retries on failure."""

from __future__ import annotations

from typing import Annotated, Any, TypeVar

from pydantic import SkipValidation
from typing_extensions import TypedDict, override

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables import Runnable, RunnableSerializable

T = TypeVar("T")

NAIVE_FIX = """Instructions:
--------------
{instructions}
--------------
Completion:
--------------
{completion}
--------------

Above, the Completion did not satisfy the constraints given in the Instructions.
Error:
--------------
{error}
--------------

Please try again. Please only respond with an answer that satisfies the constraints laid out in the Instructions:"""  # noqa: E501


NAIVE_FIX_PROMPT = PromptTemplate.from_template(NAIVE_FIX)
"""Default prompt template for fixing parsing errors."""


class OutputFixingParserRetryChainInput(TypedDict, total=False):
    """Input for the retry chain of the OutputFixingParser."""

    instructions: str
    completion: str
    error: str


class OutputFixingParser(BaseOutputParser[T]):
    """Wrap a parser and try to fix parsing errors.

    This parser wraps another output parser and, in the event that the first one
    fails to parse the output, it calls an LLM to fix the errors and try again.

    This is useful for cases where the LLM output is almost correct but has minor
    formatting issues that cause parsing to fail. Rather than failing entirely,
    this parser gives the LLM a chance to correct its mistakes.

    Example:
        .. code-block:: python

            from langchain_core.output_parsers import JsonOutputParser, OutputFixingParser
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI()
            json_parser = JsonOutputParser()

            # Wrap the parser with retry capability
            fixing_parser = OutputFixingParser.from_llm(
                llm=llm,
                parser=json_parser,
                max_retries=3
            )

            # Now parsing errors will trigger automatic retry
            result = fixing_parser.parse('{"key": value}')  # Missing quotes around value
    """

    parser: Annotated[BaseOutputParser[T], SkipValidation()]
    """The parser to use to parse the output."""

    retry_chain: Annotated[
        RunnableSerializable[OutputFixingParserRetryChainInput, str],
        SkipValidation(),
    ]
    """The runnable chain to use to retry the completion."""

    max_retries: int = 1
    """The maximum number of times to retry the parse."""

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def from_llm(
        cls,
        llm: Runnable[Any, Any],
        parser: BaseOutputParser[T],
        prompt: PromptTemplate = NAIVE_FIX_PROMPT,
        max_retries: int = 1,
    ) -> OutputFixingParser[T]:
        """Create an OutputFixingParser from a language model and a parser.

        Args:
            llm: The language model to use for fixing parsing errors.
            parser: The parser to wrap with retry capability.
            prompt: The prompt template to use for fixing errors.
                Defaults to NAIVE_FIX_PROMPT.
            max_retries: Maximum number of retries to attempt. Defaults to 1.

        Returns:
            An OutputFixingParser instance.

        Example:
            .. code-block:: python

                from langchain_core.output_parsers import (
                    PydanticOutputParser,
                    OutputFixingParser,
                )
                from langchain_openai import ChatOpenAI
                from pydantic import BaseModel

                class Person(BaseModel):
                    name: str
                    age: int

                llm = ChatOpenAI()
                parser = PydanticOutputParser(pydantic_object=Person)
                fixing_parser = OutputFixingParser.from_llm(
                    llm=llm,
                    parser=parser,
                    max_retries=3
                )
        """
        chain = prompt | llm | StrOutputParser()
        return cls(parser=parser, retry_chain=chain, max_retries=max_retries)

    @override
    def parse(self, completion: str) -> T:
        """Parse the completion, retrying with LLM if parsing fails.

        Args:
            completion: The string output from the model.

        Returns:
            The parsed output.

        Raises:
            OutputParserException: If parsing fails after all retries.
        """
        retries = 0

        while retries <= self.max_retries:
            try:
                return self.parser.parse(completion)
            except OutputParserException as e:
                if retries == self.max_retries:
                    raise
                retries += 1
                try:
                    completion = self.retry_chain.invoke(
                        {
                            "instructions": self.parser.get_format_instructions(),
                            "completion": completion,
                            "error": repr(e),
                        },
                    )
                except (NotImplementedError, AttributeError):
                    # Case: self.parser does not have get_format_instructions
                    completion = self.retry_chain.invoke(
                        {
                            "completion": completion,
                            "error": repr(e),
                        },
                    )

        msg = "Failed to parse"
        raise OutputParserException(msg)

    @override
    async def aparse(self, completion: str) -> T:
        """Async version of parse.

        Args:
            completion: The string output from the model.

        Returns:
            The parsed output.

        Raises:
            OutputParserException: If parsing fails after all retries.
        """
        retries = 0

        while retries <= self.max_retries:
            try:
                return await self.parser.aparse(completion)
            except OutputParserException as e:
                if retries == self.max_retries:
                    raise
                retries += 1
                try:
                    completion = await self.retry_chain.ainvoke(
                        {
                            "instructions": self.parser.get_format_instructions(),
                            "completion": completion,
                            "error": repr(e),
                        },
                    )
                except (NotImplementedError, AttributeError):
                    # Case: self.parser does not have get_format_instructions
                    completion = await self.retry_chain.ainvoke(
                        {
                            "completion": completion,
                            "error": repr(e),
                        },
                    )

        msg = "Failed to parse"
        raise OutputParserException(msg)

    @override
    def get_format_instructions(self) -> str:
        """Get format instructions from the wrapped parser."""
        return self.parser.get_format_instructions()

    @property
    def _type(self) -> str:
        return "output_fixing"

    @property
    @override
    def OutputType(self) -> type[T]:
        """Return the output type of the wrapped parser."""
        return self.parser.OutputType