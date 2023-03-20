from pydantic import BaseModel
from typing import Any

from langchain.guardrails.retry import naive_retry
from langchain.output_parsers import BaseOutputParser, OutputParserException
from langchain.schema import BaseLanguageModel, PromptValue


class GuardedOutputParser(BaseModel):
    """Wraps a parser and tries to fix parsing errors."""

    parser: BaseOutputParser
    retry_llm: BaseLanguageModel

    def parse(self, prompt_value: PromptValue, completion: str) -> Any:
        try:
            parsed_completion = self.parser.parse(completion)
        except OutputParserException as e:
            # TODO: can also inject str(e) into prompt, but have to validate this as a reasonable default.
            new_completion = naive_retry(
                llm=self.retry_llm, prompt=prompt_value, completion=completion,
            )
            parsed_completion = self.parser.parse(new_completion)
        
        return parsed_completion