"""Chain that gets multiple outputs from the LLM."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Extra

from langchain.chains import LLMChain, SequentialChain
from langchain.chains.base import Chain
from langchain.chains.multiple_outputs.config import VariableConfig
from langchain.chains.multiple_outputs.prompt import MultipleOutputsPrompter
from langchain.llms.base import BaseLLM
from langchain.prompts.base import DictOutputParser


class GetMultipleOutputsChain(Chain, BaseModel):
    """Chain that gets multiple outputs from the LLM, one at a time by default.

    Use this over manually parsing the outputs if the outputs can be hard to parse
    (e.g. if you're looking for code blocks of indeterminate length and content, it can
    be hard to write a regex to figure out when one block ends and the next one
    starts). It can also be used when the LLM is having difficulty generating all
    outputs in one go. The `one_step` setting lets you easily toggle between one-step
    and multi-step generation for comparison.
    """

    one_step_stop: Optional[str]
    """Stop specifically for one-step output."""
    prompter: MultipleOutputsPrompter  #: :meta private:
    # putting these two separately instead of a Union type because pydantic validation
    # fails when the LLMChain does not have values["chains"]
    one_step_chain: Optional[LLMChain]  #: :meta private:
    multi_step_chain: Optional[SequentialChain]  #: :meta private:
    completions: Optional[Dict[str, str]]  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def __init__(
        self,
        llm: BaseLLM,
        prefix: str,
        variables: Optional[Dict[str, str]] = None,
        variable_configs: Optional[List[VariableConfig]] = None,
        auto_suffix_variable_display: bool = True,
        output_parser: Optional[DictOutputParser] = None,
        one_step: bool = False,
        one_step_stop: Optional[str] = None,
        **chain_options: Any
    ):
        """Construct a chain that gets multiple outputs from the LLM.

        By default, this chain gets the inputs one at a time, although that can be
        toggled with the `one_step` variable.
        """
        prompter = MultipleOutputsPrompter(
            prefix=prefix,
            variables=variables,
            variable_configs=variable_configs,
            auto_suffix_variable_display=auto_suffix_variable_display,
            output_parser=output_parser,
        )

        one_step_chain = None
        multi_step_chain = None
        if one_step:
            one_step_chain = LLMChain(
                llm=llm,
                prompt=prompter.prompt_template_for_full_input(),
                **chain_options,
            )
        else:
            chains = []
            for i, var in enumerate(prompter.variables):
                chains.append(
                    LLMChain(
                        llm=llm,
                        prompt=prompter.prompt_template_for_variable_at(i),
                        output_key=var.output_key,
                        **chain_options,
                    )
                )
            multi_step_chain = SequentialChain(
                chains=chains,
                input_variables=[],
                output_variables=prompter.output_keys,
                **chain_options,
            )

        super().__init__(
            prompter=prompter,
            one_step_chain=one_step_chain,
            multi_step_chain=multi_step_chain,
            one_step_stop=one_step_stop,
            **chain_options,
        )

    @property
    def input_keys(self) -> List[str]:
        """Input keys for this chain.

        Always empty because the prefix should have already been pre-formatted before
        being passed into this chain.
        """
        return []

    @property
    def output_keys(self) -> List[str]:
        """Output keys produced by this whole chain."""
        return self.prompter.output_keys

    def log(self) -> str:
        """Return entire transcript for how the LLM filled in these values."""
        assert self.completions, "Run the chain first so we have something to log"
        return self.prompter.log(self.completions)

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        if self.one_step_chain:
            with_stop = {**inputs, "stop": self.one_step_stop}
            result = self.one_step_chain.apply_and_parse([with_stop])[0]
            # typechecking above should ensure a DictOutputParser gets passed in if
            # we're doing this in one step
            assert isinstance(result, dict), "Please set DictOutputParser"
            self.completions = result
        else:
            assert self.multi_step_chain is not None, "No chains configured"
            self.completions = {}
            for var, chain in zip(
                self.prompter.variables, self.multi_step_chain.chains
            ):
                assert isinstance(chain, LLMChain)
                known_values = {
                    **inputs,
                    **self.completions,
                    "stop": var.stop,
                }
                llm_result = chain.generate([known_values])
                self.completions[var.output_key] = llm_result.generations[0][0].text

        return self.completions
