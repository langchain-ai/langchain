"""Response parser."""

import json
import re

import json5

from langchain import LLMChain, PromptTemplate
from langchain.llms.base import BaseLLM
from langchain.schema import BaseOutputParser

response_template = """You are a helpful AI assistant trained to answer user queries from API responses.
You attempted to call an API, which resulted in:
API_RESPONSE: {response}

USER_COMMENT: "{instructions}"


If the API_RESPONSE can answer the USER_COMMENT respond with the following markdown json block:
Response: ```json
{{"response": "Concise response to USER_COMMENT based on API_RESPONSE."}}
```

Otherwise respond with the following markdown json block:
Response Error: ```json
{{"response": "What you did and a concise statement of the resulting error. If it can be easily fixed, provide a suggestion."}}
```

You MUST respond as a markdown json code block.

Begin:
---
"""


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
            template=response_template,
            output_parser=output_parser,
            input_variables=["response", "instructions"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
