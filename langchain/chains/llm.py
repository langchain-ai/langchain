"""Chain that just formats a prompt and calls an LLM."""
from typing import Any, Dict, List

from pydantic import BaseModel, Extra

from langchain.chains.base import Chain
from langchain.llms.base import LLM
from langchain.prompts.base import BasePrompt


class LLMChain(Chain, BaseModel):
    """Chain to run queries against LLMs.

    Example:
        .. code-block:: python

            from langchain import LLMChain, OpenAI, Prompt
            prompt_template = "Tell me a {adjective} joke"
            prompt = Prompt(input_variables=["adjective"], template=prompt_template)
            llm = LLMChain(llm=OpenAI(), prompt=prompt)
    """

    prompt: BasePrompt
    """Prompt object to use."""
    llm: LLM
    """LLM wrapper to use."""
    output_key: str = "text"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return self.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key]

    def _run(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        selected_inputs = {k: inputs[k] for k in self.prompt.input_variables}
        prompt = self.prompt.format(**selected_inputs)

        kwargs = {}
        if "stop" in inputs:
            kwargs["stop"] = inputs["stop"]
        response = self.llm(prompt, **kwargs)
        return {self.output_key: response}

    def predict(self, **kwargs: Any) -> str:
        """Format prompt with kwargs and pass to LLM.

        Args:
            **kwargs: Keys to pass to prompt template.

        Returns:
            Completion from LLM.

        Example:
            .. code-block:: python

                completion = llm.predict(adjective="funny")
        """
        return self(kwargs)[self.output_key]
