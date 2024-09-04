"""
Chain that captures the intent of the query and classifies it as a route.
Please refer to the Sineps' documentation(https://docs.sineps.io/docs/docs/guides_and_concepts/intent_router)
for more information.
"""

import os
from typing import Any, Dict, List, Optional

import sineps
from langchain.chains.base import Chain
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)


class Route:
    def __init__(self, key: str, name: str, description: str, utterances: List[str]):
        self.key = key
        self.name = name
        self.description = description
        self.utterances = utterances

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "utterances": self.utterances,
        }


class SinepsIntentRouterChain(Chain):
    """
    Chain that captures the intent of the query and classifies it as a route.
    Outputs the key of the route. If no route is found, the output key is "None".

    Example:
        .. code-block:: python

            from langchain_community.chains.sineps.intent_router import (
                SinepsIntentRouterChain,
                Route,
            )

            routes = [
                Route(key="greet", name="Greet", description="Greet the user",
                      utterances=["hello", "hi"]),
                Route(
                    key="goodbye",
                    name="Goodbye",
                    description="Say goodbye to the user",
                    utterances=["goodbye", "bye"],
                ),
            ]

            chain = SinepsIntentRouterChain(routes=routes, allow_none=True)
            output = chain({"query": "hello"})
            print(output["key"])  # Output: greet

    """

    routes: List[Route]
    allow_none: bool = False
    sineps_api_key: str = ""
    query_key: str = "query"

    @property
    def input_keys(self) -> List[str]:
        return [self.query_key]

    @property
    def output_keys(self) -> List[str]:
        return ["key"]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        query = inputs[self.query_key]
        sineps_api_key = self.sineps_api_key or os.getenv("SINEPS_API_KEY")
        client = sineps.Client(api_key=sineps_api_key)
        _routes = [route.to_dict() for route in self.routes]
        try:
            response = client.exec_intent_router(
                query=query, routes=_routes, allow_none=self.allow_none
            )
        except sineps.APIStatusError as e:
            response_text = f"{e.status_code}: {e.message}"
            _run_manager.on_text(
                response_text, color="red", end="\n", verbose=self.verbose
            )
            raise ValueError(f"Response is Invalid: {response_text}")

        if len(response.result.routes) == 0:
            return {"key": "None"}
        index = response.result.routes[0].index
        return {"key": self.routes[index].key}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        query = inputs[self.query_key]
        sineps_api_key = self.sineps_api_key or os.getenv("SINEPS_API_KEY")
        client = sineps.AsyncClient(api_key=sineps_api_key)
        _routes = [route.to_dict() for route in self.routes]
        try:
            response = await client.exec_intent_router(
                query=query, routes=_routes, allow_none=self.allow_none
            )
        except sineps.APIStatusError as e:
            response_text = f"{e.status_code}: {e.message}"
            await _run_manager.on_text(
                response_text, color="red", end="\n", verbose=self.verbose
            )
            raise ValueError(f"Response is Invalid: {response_text}")

        if len(response.result.routes) == 0:
            return {"key": "None"}
        index = response.result.routes[0].index
        return {"key": self.routes[index].key}
