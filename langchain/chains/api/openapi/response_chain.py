"""Response parser."""

import json
import re
from typing import Dict

from pydantic import root_validator

from langchain.chains.api.openapi.prompts import RESPONSE_TEMPLATE
from langchain.chains.llm import LLMChain
from langchain.llms.base import BaseLLM
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import BaseOutputParser


class APIResponderOutputParser(BaseOutputParser):
    """Parse the response and error tags."""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that json5 package exists."""
        try:
            import json5  # noqa: F401

        except ImportError:
            raise ValueError(
                "Could not import json5 python package. "
                "Please it install it with `pip install json5`."
            )
        return values

    def parse(self, llm_output: str) -> str:
        """Parse the response and error tags."""
        import json5

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
