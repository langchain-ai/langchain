"""Base classes for chain routing."""
from __future__ import annotations

from abc import ABC
from typing import Any, Dict, List, Mapping, NamedTuple, Optional

from pydantic import Extra

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
    Callbacks,
)
from langchain.chains.base import Chain


class Route(NamedTuple):
    destination: Optional[str]
    next_inputs: Dict[str, Any]


class RouterChain(Chain, ABC):
    """Chain that outputs the name of a destination chain and the inputs to it."""

    @property
    def output_keys(self) -> List[str]:
        return ["destination", "next_inputs"]

    def route(self, inputs: Dict[str, Any], callbacks: Callbacks = None) -> Route:
        result = self(inputs, callbacks=callbacks)
        return Route(result["destination"], result["next_inputs"])

    async def aroute(
        self, inputs: Dict[str, Any], callbacks: Callbacks = None
    ) -> Route:
        result = await self.acall(inputs, callbacks=callbacks)
        return Route(result["destination"], result["next_inputs"])


class MultiRouteChain(Chain):
    """Use a single chain to route an input to one of multiple candidate chains."""

    router_chain: RouterChain
    """Chain that routes inputs to destination chains."""
    destination_chains: Mapping[str, Chain]
    """Chains that return final answer to inputs."""
    default_chain: Chain
    """Default chain to use when none of the destination chains are suitable."""
    silent_errors: bool = False
    """If True, use default_chain when an invalid destination name is provided. 
    Defaults to False."""

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
        route = self.router_chain.route(inputs, callbacks=callbacks)

        _run_manager.on_text(
            str(route.destination) + ": " + str(route.next_inputs), verbose=self.verbose
        )
        if not route.destination:
            return self.default_chain(route.next_inputs, callbacks=callbacks)
        elif route.destination in self.destination_chains:
            return self.destination_chains[route.destination](
                route.next_inputs, callbacks=callbacks
            )
        elif self.silent_errors:
            return self.default_chain(route.next_inputs, callbacks=callbacks)
        else:
            raise ValueError(
                f"Received invalid destination chain name '{route.destination}'"
            )

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        route = await self.router_chain.aroute(inputs, callbacks=callbacks)

        _run_manager.on_text(
            str(route.destination) + ": " + str(route.next_inputs), verbose=self.verbose
        )
        if not route.destination:
            return await self.default_chain.acall(
                route.next_inputs, callbacks=callbacks
            )
        elif route.destination in self.destination_chains:
            return await self.destination_chains[route.destination].acall(
                route.next_inputs, callbacks=callbacks
            )
        elif self.silent_errors:
            return await self.default_chain.acall(
                route.next_inputs, callbacks=callbacks
            )
        else:
            raise ValueError(
                f"Received invalid destination chain name '{route.destination}'"
            )
