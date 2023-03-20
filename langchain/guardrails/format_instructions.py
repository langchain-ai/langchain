from __future__ import annotations

from typing import Any

from langchain.chains.llm import LLMChain
from langchain.output_parsers.base import BaseOutputParser, OutputParserException
from langchain.prompts.base import PromptValue
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import BaseLanguageModel, GuardedOutputParser

TEMPLATE = """Prompt:
{prompt}
Completion:
{completion}

Above, the Completion did not satisfy the constraints given in the Prompt.
Please try again:"""


PROMPT = PromptTemplate.from_template(TEMPLATE)


class FormatInstructionsGuard(GuardedOutputParser):
    fixer_chain: LLMChain

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel) -> FormatInstructionsGuard:
        return cls(fixer_chain=LLMChain(llm=llm, prompt=PROMPT))

    def parse(
        self, prompt_value: PromptValue, output: str, output_parser: BaseOutputParser
    ) -> Any:
        try:
            result = output_parser.parse(output)
        except OutputParserException:
            new_result = self.fixer_chain.run(
                prompt=prompt_value.to_string(), completion=output
            )
            result = output_parser.parse(new_result)
        return result
