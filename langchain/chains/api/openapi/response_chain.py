"""Response parser."""

import json
import re

import json5

from langchain import LLMChain, PromptTemplate
from langchain.chains.api.openapi.prompts import RESPONSE_TEMPLATE
from langchain.llms.base import BaseLLM
from langchain.schema import BaseOutputParser


class APIResponderOutputParser(BaseOutputParser):
    """Parse the response and error tags."""

    def parse(self, llm_output: str) -> str:
        """Parse the response and error tags."""
        json_match = re.search(r"```json(.*?)```", llm_output, re.DOTALL)
        if json_match:
            try:
                response_content = json5.loads(json_match.group(1).strip())
                return response_content.get("response", "ERROR parsing response.")
            except json.JSONDecodeError:
                return "ERROR parsing response."
            except:
                raise
        else:
            raise ValueError("No response found in output.")


class APIResponderChain(LLMChain):
    """Get the response parser."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        output_parser = APIResponderOutputParser()
        prompt = PromptTemplate(
            template=RESPONSE_TEMPLATE,
            output_parser=output_parser,
            input_variables=["response", "instructions"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
