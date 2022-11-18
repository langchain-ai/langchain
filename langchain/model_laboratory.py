"""Experiment with different models."""
from typing import List, Optional

from langchain.chains.llm import LLMChain
from langchain.input import get_color_mapping, print_text
from langchain.llms.base import LLM
from langchain.prompts.prompt import Prompt
from langchain.chains.base import Chain


class ModelLaboratory:
    """Experiment with different models."""

    def __init__(self, chains: List[Chain]):
        """Initialize with chains to experiment with.

        Args:
            chains: list of chains to experiment with.
        """
        for chain in chains:
            if len(chain.input_keys) != 1:
                raise ValueError
        self.chains = chains
        chain_range = [str(i) for i in range(len(self.chains))]
        self.chain_colors = get_color_mapping(chain_range)

    @classmethod
    def from_llms(cls, llms: List[LLM], prompt: Optional[Prompt] = None):
        """Initialize with LLMs to experiment with and optional prompt.

        Args:
            llms: list of LLMs to experiment with
            prompt: Optional prompt to use to prompt the LLMs. Defaults to None.
                If a prompt was provided, it should only have one input variable.
        """
        self.llms = llms
        llm_range = [str(i) for i in range(len(self.llms))]
        self.llm_colors = get_color_mapping(llm_range)
        if prompt is None:
            self.prompt = Prompt(input_variables=["_input"], template="{_input}")
        else:
            if len(prompt.input_variables) != 1:
                raise ValueError(
                    "Currently only support prompts with one input variable, "
                    f"got {prompt}"
                )
            self.prompt = prompt

    def compare(self, text: str) -> None:
        """Compare model outputs on an input text.

        If a prompt was provided with starting the laboratory, then this text will be
        fed into the prompt. If no prompt was provided, then the input text is the
        entire prompt.

        Args:
            text: input text to run all models on.
        """
        print(f"\033[1mInput:\033[0m\n{text}\n")
        for i, chain in enumerate(self.chains):
            print_text(str(chain), end="\n")
            inputs = {self.prompt.input_variables[0]: text}
            output = chain(inputs)
            if len(output) == 1:
                str_output = list(output.values())[0]
            else:
                str_output = str(output)
            print_text(str_output, color=self.llm_colors[str(i)], end="\n\n")
