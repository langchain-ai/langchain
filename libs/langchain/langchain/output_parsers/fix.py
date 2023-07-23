from __future__ import annotations

from typing import TypeVar

from langchain.chains.llm import LLMChain
from langchain.output_parsers.prompts import NAIVE_FIX_PROMPT
from langchain.schema import BaseOutputParser, BasePromptTemplate, OutputParserException
from langchain.schema.language_model import BaseLanguageModel

T = TypeVar("T")


class OutputFixingParser(BaseOutputParser[T]):
    """Wraps a parser and tries to fix parsing errors."""

    @property
    def lc_serializable(self) -> bool:
        return True

    parser: BaseOutputParser[T]
    retry_chain: LLMChain

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        parser: BaseOutputParser[T],
        prompt: BasePromptTemplate = NAIVE_FIX_PROMPT,
    ) -> OutputFixingParser[T]:
        """Create an OutputFixingParser from a language model and a parser.

        Args:
            llm: llm to use for fixing
            parser: parser to use for parsing
            prompt: prompt to use for fixing

        Returns:
            OutputFixingParser
        """
        chain = LLMChain(llm=llm, prompt=prompt)
        return cls(parser=parser, retry_chain=chain)

    def parse(self, completion: str) -> T:
        try:
            parsed_completion = self.parser.parse(completion)
        except OutputParserException as e:
            new_completion = self.retry_chain.run(
                instructions=self.parser.get_format_instructions(),
                completion=completion,
                error=repr(e),
            )
            parsed_completion = self.parser.parse(new_completion)

        return parsed_completion

    def get_format_instructions(self) -> str:
        return self.parser.get_format_instructions()

    @property
    def _type(self) -> str:
        return "output_fixing"
