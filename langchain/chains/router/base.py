"""Base classes for chain routing."""
from __future__ import annotations

from abc import ABC
from typing import Any, Dict, List, Mapping, Optional

from pydantic import Extra

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain


class RouterChain(Chain, ABC):
    """Chain that outputs the name of a destination chain and the inputs to it."""

    @property
    def output_keys(self) -> List[str]:
        return ["destination", "next_inputs"]


class MultiRouteChain(Chain):
    """Use a single chain to route an input to one of multiple candidate chains."""

    router_chain: RouterChain
    """Chain that routes inputs to destination chains."""
    destination_chains: Mapping[str, Chain]
    """Chains that return final answer to inputs."""
    default_chain: Chain
    """Default chain to use when routing fails."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the router chain prompt expects.

        :meta private:
        """
        return self.router_chain.input_keys

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return []

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        router_output = self.router_chain(inputs, callbacks=callbacks)
        destination = router_output["destination"]
        next_inputs = router_output["next_inputs"]
        _run_manager.on_text(
            destination + ": " + str(next_inputs), verbose=self.verbose
        )
        if destination in self.destination_chains:
            return self.destination_chains[destination](
                next_inputs, callbacks=callbacks
            )
        else:
            return self.default_chain(next_inputs, callbacks=callbacks)
