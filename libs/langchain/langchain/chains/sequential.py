"""Chain pipeline where the outputs of one step feed directly into next."""

from typing import Any, Dict, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.utils.input import get_color_mapping
from pydantic import ConfigDict, model_validator
from typing_extensions import Self

from langchain.chains.base import Chain


class SequentialChain(Chain):
    """Chain where the outputs of one chain feed directly into next."""

    chains: List[Chain]
    input_variables: List[str]
    output_variables: List[str]  #: :meta private:
    return_all: bool = False

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @property
    def input_keys(self) -> List[str]:
        """Return expected input keys to the chain.

        :meta private:
        """
        return self.input_variables

    @property
    def output_keys(self) -> List[str]:
        """Return output key.

        :meta private:
        """
        return self.output_variables

    @model_validator(mode="before")
    @classmethod
    def validate_chains(cls, values: Dict) -> Any:
        """Validate that the correct inputs exist for all chains."""
        chains = values["chains"]
        input_variables = values["input_variables"]
        memory_keys = list()
        if "memory" in values and values["memory"] is not None:
            """Validate that prompt input variables are consistent."""
            memory_keys = values["memory"].memory_variables
            if set(input_variables).intersection(set(memory_keys)):
                overlapping_keys = set(input_variables) & set(memory_keys)
                raise ValueError(
                    f"The input key(s) {''.join(overlapping_keys)} are found "
                    f"in the Memory keys ({memory_keys}) - please use input and "
                    f"memory keys that don't overlap."
                )

        known_variables = set(input_variables + memory_keys)

        for chain in chains:
            missing_vars = set(chain.input_keys).difference(known_variables)
            if chain.memory:
                missing_vars = missing_vars.difference(chain.memory.memory_variables)

            if missing_vars:
                raise ValueError(
                    f"Missing required input keys: {missing_vars}, "
                    f"only had {known_variables}"
                )
            overlapping_keys = known_variables.intersection(chain.output_keys)
            if overlapping_keys:
                raise ValueError(
                    f"Chain returned keys that already exist: {overlapping_keys}"
                )

            known_variables |= set(chain.output_keys)

        if "output_variables" not in values:
            if values.get("return_all", False):
                output_keys = known_variables.difference(input_variables)
            else:
                output_keys = chains[-1].output_keys
            values["output_variables"] = output_keys
        else:
            missing_vars = set(values["output_variables"]).difference(known_variables)
            if missing_vars:
                raise ValueError(
                    f"Expected output variables that were not found: {missing_vars}."
                )

        return values

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        known_values = inputs.copy()
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        for i, chain in enumerate(self.chains):
            callbacks = _run_manager.get_child()
            outputs = chain(known_values, return_only_outputs=True, callbacks=callbacks)
            known_values.update(outputs)
        return {k: known_values[k] for k in self.output_variables}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        known_values = inputs.copy()
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        for i, chain in enumerate(self.chains):
            outputs = await chain.acall(
                known_values, return_only_outputs=True, callbacks=callbacks
            )
            known_values.update(outputs)
        return {k: known_values[k] for k in self.output_variables}


class SimpleSequentialChain(Chain):
    """Simple chain where the outputs of one step feed directly into next."""

    chains: List[Chain]
    strip_outputs: bool = False
    input_key: str = "input"  #: :meta private:
    output_key: str = "output"  #: :meta private:

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return output key.

        :meta private:
        """
        return [self.output_key]

    @model_validator(mode="after")
    def validate_chains(self) -> Self:
        """Validate that chains are all single input/output."""
        for chain in self.chains:
            if len(chain.input_keys) != 1:
                raise ValueError(
                    "Chains used in SimplePipeline should all have one input, got "
                    f"{chain} with {len(chain.input_keys)} inputs."
                )
            if len(chain.output_keys) != 1:
                raise ValueError(
                    "Chains used in SimplePipeline should all have one output, got "
                    f"{chain} with {len(chain.output_keys)} outputs."
                )
        return self

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        _input = inputs[self.input_key]
        color_mapping = get_color_mapping([str(i) for i in range(len(self.chains))])
        for i, chain in enumerate(self.chains):
            _input = chain.run(
                _input, callbacks=_run_manager.get_child(f"step_{i + 1}")
            )
            if self.strip_outputs:
                _input = _input.strip()
            _run_manager.on_text(
                _input, color=color_mapping[str(i)], end="\n", verbose=self.verbose
            )
        return {self.output_key: _input}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        _input = inputs[self.input_key]
        color_mapping = get_color_mapping([str(i) for i in range(len(self.chains))])
        for i, chain in enumerate(self.chains):
            _input = await chain.arun(
                _input, callbacks=_run_manager.get_child(f"step_{i + 1}")
            )
            if self.strip_outputs:
                _input = _input.strip()
            await _run_manager.on_text(
                _input, color=color_mapping[str(i)], end="\n", verbose=self.verbose
            )
        return {self.output_key: _input}
