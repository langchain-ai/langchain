"""Chain that just formats a prompt and calls an LLM."""
from typing import Any, Dict, List

from pydantic import BaseModel, Extra

from langchain.chains.base import Chain
from langchain.llms.base import LLM
from langchain.prompt import Prompt


class LLMChain(Chain, BaseModel):
    """Chain to run queries against LLMs."""

    prompt: Prompt
    llm: LLM
    return_key: str = "text"

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects."""
        return self.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key."""
        return [self.return_key]

    def _run(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        selected_inputs = {k: inputs[k] for k in self.prompt.input_variables}
        prompt = self.prompt.template.format(**selected_inputs)

        kwargs = {}
        if "stop" in inputs:
            kwargs["stop"] = inputs["stop"]
        response = self.llm(prompt, **kwargs)
        return {self.return_key: response}

    def predict(self, **kwargs: Any) -> str:
        """More user-friendly interface for interacting with LLMs."""
        return self(kwargs)[self.return_key]
