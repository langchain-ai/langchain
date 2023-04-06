"""request parser."""

import json
import re

import json5

from langchain.chains.api.openapi.prompts import REQUEST_TEMPLATE
from langchain.chains.llm import LLMChain
from langchain.llms.base import BaseLLM
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import BaseOutputParser


class APIRequesterOutputParser(BaseOutputParser):
    """Parse the request and error tags."""

    def parse(self, llm_output: str) -> str:
        """Parse the request and error tags."""
        json_match = re.search(r"```json(.*?)```", llm_output, re.DOTALL)
        if json_match:
            typescript_block = json_match.group(1).strip()
            try:
                return json.dumps(json5.loads(typescript_block))
            except json.JSONDecodeError:
                return "ERROR serializing request"
        message_match = re.search(r"```text(.*?)```", llm_output, re.DOTALL)
        if message_match:
            return f"MESSAGE: {message_match.group(1).strip()}"
        return "ERROR making request"


class APIRequesterChain(LLMChain):
    """Get the request parser."""

    @classmethod
    def from_llm_and_typescript(
        cls, llm: BaseLLM, typescript_definition: str, verbose: bool = True
    ) -> LLMChain:
        """Get the request parser."""
        output_parser = APIRequesterOutputParser()
        prompt = PromptTemplate(
            template=REQUEST_TEMPLATE,
            output_parser=output_parser,
            partial_variables={"schema": typescript_definition},
            input_variables=["instructions"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
