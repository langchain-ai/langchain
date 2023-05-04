from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain.chains.base import Chain
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks.stdout import StdOutCallbackHandler


class MapChain(Chain):
    """ This chain operates on a list of input values (input_values) and maps them using the mapper_chain
    The functionality is similar to map() in functional programming
    """

    # input/output keys
    input_values_key: str = "input_values"
    output_values_keys: str = "output_values"

    mapper_chain: Chain

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return [self.input_values_key]

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_values_keys]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:

        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        input_values = inputs[self.input_values_key]

        other_keys: Dict = {k: v for k, v in inputs.items() if k not in self.input_keys}

        output_values = []
        for input in input_values:

            other_keys["input"] = input
            output = self.mapper_chain(other_keys, callbacks=_run_manager.get_child())
            output_value = output["output"]
            output_values.append(output_value)

        return {
            self.output_values_keys: output_values
        }

    @property
    def _chain_type(self) -> str:
        return "MapChain"
