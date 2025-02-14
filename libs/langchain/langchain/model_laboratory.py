"""Experiment with different models."""

from __future__ import annotations

from typing import List, Optional, Sequence

from langchain_core.language_models.llms import BaseLLM
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.utils.input import get_color_mapping, print_text

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain


class ModelLaboratory:
    """A utility to experiment with and compare the performance of different models."""

    def __init__(self, chains: Sequence[Chain], names: Optional[List[str]] = None):
        """Initialize the ModelLaboratory with chains to experiment with.

        Args:
            chains (Sequence[Chain]): A sequence of chains to experiment with.
            Each chain must have exactly one input and one output variable.
        names (Optional[List[str]]): Optional list of names corresponding to each chain.
            If provided, its length must match the number of chains.


        Raises:
            ValueError: If any chain is not an instance of `Chain`.
            ValueError: If a chain does not have exactly one input variable.
            ValueError: If a chain does not have exactly one output variable.
            ValueError: If the length of `names` does not match the number of chains.
        """
        for chain in chains:
            if not isinstance(chain, Chain):
                raise ValueError(
                    "ModelLaboratory should now be initialized with Chains. "
                    "If you want to initialize with LLMs, use the `from_llms` method "
                    "instead (`ModelLaboratory.from_llms(...)`)"
                )
            if len(chain.input_keys) != 1:
                raise ValueError(
                    "Currently only support chains with one input variable, "
                    f"got {chain.input_keys}"
                )
            if len(chain.output_keys) != 1:
                raise ValueError(
                    "Currently only support chains with one output variable, "
                    f"got {chain.output_keys}"
                )
        if names is not None:
            if len(names) != len(chains):
                raise ValueError("Length of chains does not match length of names.")
        self.chains = chains
        chain_range = [str(i) for i in range(len(self.chains))]
        self.chain_colors = get_color_mapping(chain_range)
        self.names = names

    @classmethod
    def from_llms(
        cls, llms: List[BaseLLM], prompt: Optional[PromptTemplate] = None
    ) -> ModelLaboratory:
        """Initialize the ModelLaboratory with LLMs and an optional prompt.

        Args:
            llms (List[BaseLLM]): A list of LLMs to experiment with.
            prompt (Optional[PromptTemplate]): An optional prompt to use with the LLMs.
                If provided, the prompt must contain exactly one input variable.

        Returns:
            ModelLaboratory: An instance of `ModelLaboratory` initialized with LLMs.
        """
        if prompt is None:
            prompt = PromptTemplate(input_variables=["_input"], template="{_input}")
        chains = [LLMChain(llm=llm, prompt=prompt) for llm in llms]
        names = [str(llm) for llm in llms]
        return cls(chains, names=names)

    def compare(self, text: str) -> None:
        """Compare model outputs on an input text.

        If a prompt was provided with starting the laboratory, then this text will be
        fed into the prompt. If no prompt was provided, then the input text is the
        entire prompt.

        Args:
            text: input text to run all models on.
        """
        print(f"\033[1mInput:\033[0m\n{text}\n")  # noqa: T201
        for i, chain in enumerate(self.chains):
            if self.names is not None:
                name = self.names[i]
            else:
                name = str(chain)
            print_text(name, end="\n")
            output = chain.run(text)
            print_text(output, color=self.chain_colors[str(i)], end="\n\n")
