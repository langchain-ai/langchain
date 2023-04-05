"""request parser."""

import json
import re
from typing import Union

import json5
from openapi_schema_pydantic import OpenAPI, Schema

from langchain import LLMChain, PromptTemplate
from langchain.chains.api.openapi.typescript_utils import schema_to_typescript
from langchain.llms import BaseLLM
from langchain.schema import BaseOutputParser

request_template = """You are a helpful AI Assistant. Please provide JSON arguments to agentFunc() based on the user's instructions.

API_SCHEMA: ```typescript
type agentFunc = (_: {schema}
) => any;
```

USER_INSTRUCTIONS: "{instructions}"

Your arguments must be plain json provided in a markdown block:

ARGS: ```json
{{valid json conforming to API_SCHEMA}}
```

Example
-----

ARGS: ```json
{{"foo": "bar", "baz": {{"qux": "quux"}}}}
```

The block must be no more than 1 line long, and all arguments must be valid JSON. All string arguments must be wrapped in double quotes.
You MUST strictly comply to the types indicated by the provided schema, including all required args.

If you don't have sufficient information to call the function due to things like requiring specific uuid's, you can reply with the following message:

Message: ```text
Concise response requesting the additional information that would make calling the function successful.
```

Begin
-----
ARGS:
"""


class APIRequesterOutputParser(BaseOutputParser):
    """Parse the request and error tags."""

    def parse(self, llm_output: str) -> Union[dict, str]:
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
            template=request_template,
            output_parser=output_parser,
            partial_variables={"schema": typescript_definition},
            input_variables=["instructions"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
