"""Response parser."""

import json
import re
from typing import Any

from langchain.chains.api.openapi.prompts import RESPONSE_TEMPLATE
from langchain.chains.llm import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import BaseOutputParser
from langchain.schema.language_model import BaseLanguageModel


class APIResponderOutputParser(BaseOutputParser):
    """Parse the response and error tags."""

    def _load_json_block(self, serialized_block: str) -> str:
        try:
            response_content = json.loads(serialized_block, strict=False)
            return response_content.get("response", "ERROR parsing response.")
        except json.JSONDecodeError:
            return "ERROR parsing response."
        except:
            raise

    def parse(self, llm_output: str) -> str:
        """Parse the response and error tags."""
        json_match = re.search(r"```json(.*?)```", llm_output, re.DOTALL)
        if json_match:
            return self._load_json_block(json_match.group(1).strip())
        else:
            raise ValueError(f"No response found in output: {llm_output}.")

    @property
    def _type(self) -> str:
        return "api_responder"


class APIResponderChain(LLMChain):
    """Get the response parser."""

    @classmethod
    def from_llm(
        cls, llm: BaseLanguageModel, verbose: bool = True, **kwargs: Any
    ) -> LLMChain:
        """Get the response parser."""
        output_parser = APIResponderOutputParser()
        prompt = PromptTemplate(
            template=RESPONSE_TEMPLATE,
            output_parser=output_parser,
            input_variables=["response", "instructions"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose, **kwargs)
