from __future__ import annotations

from typing import Annotated, Any, TypeVar

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import Runnable, RunnableSerializable
from pydantic import SkipValidation
from typing_extensions import TypedDict, override

from langchain_classic.output_parsers.prompts import NAIVE_FIX_PROMPT

T = TypeVar("T")


class OutputFixingParserRetryChainInput(TypedDict, total=False):
    """Input for the retry chain of the OutputFixingParser."""

    instructions: str
    completion: str
    error: str


class OutputFixingParser(BaseOutputParser[T]):
    """Wrap a parser and try to fix parsing errors."""

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        return True

    parser: Annotated[Any, SkipValidation()]
    """The parser to use to parse the output."""
    # Should be an LLMChain but we want to avoid top-level imports from
    # langchain_classic.chains
    retry_chain: Annotated[
        RunnableSerializable[OutputFixingParserRetryChainInput, str] | Any,
        SkipValidation(),
    ]
    """The RunnableSerializable to use to retry the completion (Legacy: LLMChain)."""
    max_retries: int = 1
    """The maximum number of times to retry the parse."""
    legacy: bool = True
    """Whether to use the run or arun method of the retry_chain."""

    @classmethod
    def from_llm(
        cls,
        llm: Runnable,
        parser: BaseOutputParser[T],
        prompt: BasePromptTemplate = NAIVE_FIX_PROMPT,
        max_retries: int = 1,
    ) -> OutputFixingParser[T]:
        """Create an OutputFixingParser from a language model and a parser.

        Args:
            llm: llm to use for fixing
            parser: parser to use for parsing
            prompt: prompt to use for fixing
            max_retries: Maximum number of retries to parse.

        Returns:
            OutputFixingParser
        """
        chain = prompt | llm | StrOutputParser()
        return cls(parser=parser, retry_chain=chain, max_retries=max_retries)

    @override
    def parse(self, completion: str) -> T:
        retries = 0

        while retries <= self.max_retries:
            try:
                return self.parser.parse(completion)
            except OutputParserException as e:
                if retries == self.max_retries:
                    raise
                retries += 1
                if self.legacy and hasattr(self.retry_chain, "run"):
                    completion = self.retry_chain.run(
                        instructions=self.parser.get_format_instructions(),
                        completion=completion,
                        error=repr(e),
                    )
                else:
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
        retries = 0

        while retries <= self.max_retries:
            try:
                return await self.parser.aparse(completion)
            except OutputParserException as e:
                if retries == self.max_retries:
                    raise
                retries += 1
                if self.legacy and hasattr(self.retry_chain, "arun"):
                    completion = await self.retry_chain.arun(
                        instructions=self.parser.get_format_instructions(),
                        completion=completion,
                        error=repr(e),
                    )
                else:
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
        return self.parser.get_format_instructions()

    @property
    def _type(self) -> str:
        return "output_fixing"

    @property
    @override
    def OutputType(self) -> type[T]:
        return self.parser.OutputType
