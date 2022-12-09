"""Chain that just formats a prompt and calls an LLM."""
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Extra

from langchain.chains.base import Chain
from langchain.input import print_text
from langchain.llms.base import LLM
from langchain.prompts.base import BasePromptTemplate

import itertools


class LLMChain(Chain, BaseModel):
    """Chain to run queries against LLMs.

    Example:
        .. code-block:: python

            from langchain import LLMChain, OpenAI, PromptTemplate
            prompt_template = "Tell me a {adjective} joke"
            prompt = PromptTemplate(
                input_variables=["adjective"], template=prompt_template
            )
            llm = LLMChain(llm=OpenAI(), prompt=prompt)
    """

    prompt: BasePromptTemplate
    """Prompt object to use."""
    llm: LLM
    """LLM wrapper to use."""
    n: int = 1
    """Fanout"""
    max_branching_factor: int = 5
    """Max number of parallel children"""
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

    def _call(self, inputs: Dict[str, Any], n: Optional[int] = 1) -> Dict[str, Any]:
        selected_inputs = {k: inputs[k] for k in self.prompt.input_variables}
        print("selected_inputs")
        print(selected_inputs)

        def listify(i):
            if isinstance(i, str):
                return [i]
            else:
                return i

        list_of_selected_inputs = [
            listify(inputs[k]) for k in self.prompt.input_variables
        ]
        print("list_of_selected_inputs")
        print(list_of_selected_inputs)
        list_of_branches = []

        for options in itertools.product(*list_of_selected_inputs):
            # print("options")
            # print(options)
            list_of_branches.append(
                {k: v for k, v in zip(self.prompt.input_variables, options)}
            )
        print("list_of_branches:")
        print(list_of_branches)
        # print(list_of_branches)
        # selected_inputs

        # issue branching calls
        responses = []
        # for branch in list_of_branches:

        def branch_call(branch):
            prompt = self.prompt.format(**selected_inputs)
            if self.verbose:
                print("Prompt after formatting:")
                print_text(prompt, color="green", end="\n")
            kwargs = {}
            if "stop" in inputs:
                kwargs["stop"] = inputs["stop"]
            # print("hi")
            # print(n)
            # print(self.n)
            response = self.llm(prompt, n=self.n or n, **kwargs)
            return response

        executor = ThreadPoolExecutor(self.max_branching_factor)
        responses = executor.map(branch_call, list_of_branches)
        responses = list(itertools.chain(*responses))
        print("responses")
        print(responses)
        return {self.output_key: responses}

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

    def predict_and_parse(self, **kwargs: Any) -> Union[str, List[str], Dict[str, str]]:
        """Call predict and then parse the results."""
        result = self.predict(**kwargs)
        if self.prompt.output_parser is not None:
            return self.prompt.output_parser.parse(result)
        else:
            return result
