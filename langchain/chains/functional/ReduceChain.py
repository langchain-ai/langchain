from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain.chains.base import Chain
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)

from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks.stdout import StdOutCallbackHandler


class ReduceChain(Chain):
    """ This chain operates on a list of input values (input_values)
    Each item in the input_values is passed to the reducer_chain with two keys accumulated_value, current_value
    The behaviour is similar to reduce() in functional programming
    """

    # input/output keys
    input_values_key: str = "input_values"
    initial_value_key: str = "initial_value"
    output_value_key: str = "output_value"

    reducer_chain: Chain

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return [self.input_values_key, self.initial_value_key]

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_value_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:

        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        input_values = inputs[self.input_values_key]
        initial_value = inputs[self.initial_value_key]

        accumulated_value = initial_value

        other_keys: Dict = {k: v for k, v in inputs.items() if k not in self.input_keys}

        for current_value in input_values:

            other_keys["accumulated_value"] = accumulated_value
            other_keys["current_value"] = current_value

            output = self.reducer_chain(other_keys, callbacks=_run_manager.get_child())
            accumulated_value = output["output_value"]

        return {
            self.output_value_key: accumulated_value
        }

    @property
    def _chain_type(self) -> str:
        return "ReduceChain"
