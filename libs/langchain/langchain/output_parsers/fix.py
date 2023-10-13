from __future__ import annotations

from typing import Any, TypeVar

from langchain.output_parsers.prompts import NAIVE_FIX_PROMPT
from langchain.schema import BaseOutputParser, BasePromptTemplate, OutputParserException
from langchain.schema.language_model import BaseLanguageModel

T = TypeVar("T")


class OutputFixingParser(BaseOutputParser[T]):
    """Wraps a parser and tries to fix parsing errors."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    parser: BaseOutputParser[T]
    # Should be an LLMChain but we want to avoid top-level imports from langchain.chains
    retry_chain: Any
    max_retries: int = 1

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        parser: BaseOutputParser[T],
        prompt: BasePromptTemplate = NAIVE_FIX_PROMPT,
        max_retries: int = 1,
    ) -> OutputFixingParser[T]:
        """Create an OutputFixingParser from a language model and a parser.

        Args:
            llm: llm to use for fixing
            parser: parser to use for parsing
            prompt: prompt to use for fixing
            max_retries: Maximum number of retries to parser.

        Returns:
            OutputFixingParser
        """
        from langchain.chains.llm import LLMChain

        chain = LLMChain(llm=llm, prompt=prompt)
        return cls(parser=parser, retry_chain=chain, max_retries=max_retries)

    def parse(self, completion: str) -> T:
        retries = 0

        while retries <= self.max_retries:
            try:
                return self.parser.parse(completion)
            except OutputParserException as e:
                if retries == self.max_retries:
                    raise e
                else:
                    retries += 1
                    completion = self.retry_chain.run(
                        instructions=self.parser.get_format_instructions(),
                        completion=completion,
                        error=repr(e),
                    )

        raise OutputParserException("Failed to parse")

    async def aparse(self, completion: str) -> T:
        retries = 0

        while retries <= self.max_retries:
            try:
                return await self.parser.aparse(completion)
            except OutputParserException as e:
                if retries == self.max_retries:
                    raise e
                else:
                    retries += 1
                    completion = await self.retry_chain.arun(
                        instructions=self.parser.get_format_instructions(),
                        completion=completion,
                        error=repr(e),
                    )

        raise OutputParserException("Failed to parse")

    def get_format_instructions(self) -> str:
        return self.parser.get_format_instructions()

    @property
    def _type(self) -> str:
        return "output_fixing"
