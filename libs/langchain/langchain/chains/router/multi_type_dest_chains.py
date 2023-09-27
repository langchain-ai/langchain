from langchain.chains.router.base import MultiRouteChain, RouterChain
from typing import Any, Dict, List, Mapping, Optional
from langchain.chains.base import Chain
from langchain.chains import LLMChain

class MultiTypeDestRouteChain(MultiRouteChain):
    """A multi-route chain that uses and LLM router chain to choose amongst prompts."""
    router_chain: RouterChain
    """chain for decisding a destination chain and the input to it."""
    
    destination_chains: Mapping[str,Chain]
    """Map of name to candidate chains that inputs can be routed to."""
    default_chain: LLMChain
    """Default chain to use when router doesn't map input to one of the destinations"""

    @property
    def output_keys(self) -> List[str]:
        return["text"]