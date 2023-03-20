from typing import Any

from langchain.guardrails.retry import naive_retry
from langchain.output_parsers import BaseOutputParser, OutputParserException
from langchain.schema import BaseLanguageModel, PromptValue


class RetriableOutputParser(BaseOutputParser):
    """Wraps a parser and tries to fix parsing errors."""

    parser: BaseOutputParser
    retry_llm: BaseLanguageModel

    def parse_with_prompt(self, completion: str, prompt_value: PromptValue) -> Any:
        try:
            parsed_completion = self.parser.parse(completion)
        except OutputParserException as e:
            # TODO: can also inject str(e) into prompt, but have to validate this as a reasonable default.
            new_completion = naive_retry(
                llm=self.retry_llm, prompt=prompt_value, completion=completion,
            )
            parsed_completion = self.parser.parse(new_completion)
        
        return parsed_completion

    def parse(self, completion: str):
        raise NotImplementedError

    def get_format_instructions(self) -> str:
        return self.parser.get_format_instructions()
