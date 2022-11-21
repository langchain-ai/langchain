"""Load routing chains."""
from typing import Any, List

from langchain.llms.base import LLM
from langchain.routing_chains.mrkl.base import ZeroShotRouter
from langchain.routing_chains.react.base import ReActDocstoreRouter
from langchain.routing_chains.routing_chain import RoutingChain
from langchain.routing_chains.self_ask_with_search.base import SelfAskWithSearchRouter
from langchain.routing_chains.tools import Tool

ROUTER_TYPE_TO_CLASS = {
    "zero-shot-react-description": ZeroShotRouter,
    "react-docstore": ReActDocstoreRouter,
    "self-ask-with-search": SelfAskWithSearchRouter,
}


def load_routing_chain(
    tools: List[Tool],
    llm: LLM,
    router_type: str = "zero-shot-react-description",
    **kwargs: Any,
) -> RoutingChain:
    """Load routing chain given tools and LLM.

    Args:
        tools: List of tools this routing chain has access to.
        llm: Language model to use as the router.
        router_type: The router to use. Valid options are:
            `zero-shot-react-description`.
        **kwargs: Additional key word arguments to pass to the routing chain.

    Returns:
        A routing chain.
    """
    if router_type not in ROUTER_TYPE_TO_CLASS:
        raise ValueError(
            f"Got unknown router type: {router_type}. "
            f"Valid types are: {ROUTER_TYPE_TO_CLASS.keys()}."
        )
    router_cls = ROUTER_TYPE_TO_CLASS[router_type]
    router = router_cls.from_llm_and_tools(llm, tools)
    return RoutingChain(router=router, tools=tools, **kwargs)
