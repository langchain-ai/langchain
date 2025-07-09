"""Base classes for chain routing."""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
from typing import Any, NamedTuple, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
    Callbacks,
)
from pydantic import ConfigDict

from langchain.chains.base import Chain


class Route(NamedTuple):
    destination: Optional[str]
    next_inputs: dict[str, Any]


class RouterChain(Chain, ABC):
    """Chain that outputs the name of a destination chain and the inputs to it."""

    @property
    def output_keys(self) -> list[str]:
        return ["destination", "next_inputs"]

    def route(self, inputs: dict[str, Any], callbacks: Callbacks = None) -> Route:
        """
        Route inputs to a destination chain.

        Args:
            inputs: inputs to the chain
            callbacks: callbacks to use for the chain

        Returns:
            a Route object
        """
        result = self(inputs, callbacks=callbacks)
        return Route(result["destination"], result["next_inputs"])

    async def aroute(
        self,
        inputs: dict[str, Any],
        callbacks: Callbacks = None,
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

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @property
    def input_keys(self) -> list[str]:
        """Will be whatever keys the router chain prompt expects.

        :meta private:
        """
        return self.router_chain.input_keys

    @property
    def output_keys(self) -> list[str]:
        """Will always return text key.

        :meta private:
        """
        return []

    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        route = self.router_chain.route(inputs, callbacks=callbacks)

        _run_manager.on_text(
            str(route.destination) + ": " + str(route.next_inputs),
            verbose=self.verbose,
        )
        if not route.destination:
            return self.default_chain(route.next_inputs, callbacks=callbacks)
        if route.destination in self.destination_chains:
            return self.destination_chains[route.destination](
                route.next_inputs,
                callbacks=callbacks,
            )
        if self.silent_errors:
            return self.default_chain(route.next_inputs, callbacks=callbacks)
        msg = f"Received invalid destination chain name '{route.destination}'"
        raise ValueError(msg)

    async def _acall(
        self,
        inputs: dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> dict[str, Any]:
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        route = await self.router_chain.aroute(inputs, callbacks=callbacks)

        await _run_manager.on_text(
            str(route.destination) + ": " + str(route.next_inputs),
            verbose=self.verbose,
        )
        if not route.destination:
            return await self.default_chain.acall(
                route.next_inputs,
                callbacks=callbacks,
            )
        if route.destination in self.destination_chains:
            return await self.destination_chains[route.destination].acall(
                route.next_inputs,
                callbacks=callbacks,
            )
        if self.silent_errors:
            return await self.default_chain.acall(
                route.next_inputs,
                callbacks=callbacks,
            )
        msg = f"Received invalid destination chain name '{route.destination}'"
        raise ValueError(msg)
