"""Chain that just formats a prompt and calls an LLM."""
from typing import Any, Dict, List

from pydantic import BaseModel, Extra

from langchain.chains.base import Chain
from langchain.docstore.base import Docstore
from langchain.input import print_text
from langchain.llms.base import LLM
from langchain.prompts.base import BasePromptTemplate


class MemoryChain(Chain, BaseModel):
    """Chain to run queries against LLMs and save intermediate reuslts in a docstore.

    Example:
        .. code-block:: python

            from langchain import LLMChain, OpenAI, PromptTemplate, InMemoryDocstore
            prompt_template = "Talk to me!\n\n{history}\nHuman: {input}\nAI:"
            docstore = InMemoryDocstore()
            prompt = PromptTemplate(
                input_variables=["input"], template=prompt_template
            )
            memory_chain = MemoryChain(
                llm=OpenAI(),
                docstore=docstore,
                prompt=prompt
            )
    """

    prompt: BasePromptTemplate
    """Prompt object to use."""
    llm: LLM
    """LLM wrapper to use."""
    docstore: Docstore
    """Docstore to use."""
    history_key: str = "history"
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

    def _fetch_history_from_docstore(self, inputs):
        return self.docstore.list()

    def _format_inputs_for_docstore(self, inputs):
        return '\n'.join(["{k}: {v}".format(k=k, v=v) for k, v in inputs.items()])

    def _format_output_for_docstore(self, output):
        return "{output}".format(output)

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        selected_inputs = {k: inputs[k] for k in self.prompt.input_variables}
        history = self._fetch_history_from_docstore(selected_inputs)
        selected_inputs[self.history_key] = history
        prompt = self.prompt.format(**selected_inputs)
        if self.verbose:
            print("Prompt after formatting:")
            print_text(prompt, color="green", end="\n")
        kwargs = {}
        if "stop" in inputs:
            kwargs["stop"] = inputs["stop"]
        response = self.llm(prompt, **kwargs)
        inputs = self._format_inputs_for_docstore(selected_inputs)
        output = self._format_output_for_docstore(response)
        self.docstore.add([inputs, output])
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
