"""Use a single chain to route an input to one of multiple llm chains."""
from __future__ import annotations

from typing import Any, List, Mapping, Optional

from langchain.base_language import BaseLanguageModel
from langchain.chains import ConversationChain
from langchain.chains.llm import LLMChain
from langchain.chains.router.base import MultiRouteChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate


class MultiPromptChain(MultiRouteChain):
    """A multi-route chain that uses an LLM router chain to choose amongst prompts."""

    router_chain: LLMRouterChain
    """Chain for deciding a destination chain and the input to it."""
    destination_chains: Mapping[str, LLMChain]
    """Map of name to candidate chains that inputs can be routed to."""
    default_chain: LLMChain
    """Default chain to use when router doesn't map input to one of the destinations."""

    @property
    def output_keys(self) -> List[str]:
        return ["text"]

    @classmethod
    def from_prompts(
        cls,
        llm: BaseLanguageModel,
        prompt_names: List[str],
        prompt_descriptions: List[str],
        prompt_templates: List[str],
        default_chain: Optional[LLMChain] = None,
        **kwargs: Any,
    ) -> MultiPromptChain:
        """Convenience constructor for instantiating from destination prompts."""
        destinations = [
            f"{name}: {description}"
            for name, description in zip(prompt_names, prompt_descriptions)
        ]
        destinations_str = "\n".join(destinations)
        router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
            destinations=destinations_str
        )
        router_prompt = PromptTemplate(
            template=router_template,
            input_variables=["input"],
            output_parser=RouterOutputParser(),
        )
        router_chain = LLMRouterChain.from_llm(llm, router_prompt)
        destination_chains = {
            name: LLMChain(
                llm=llm,
                prompt=PromptTemplate(template=prompt, input_variables=["input"]),
            )
            for name, prompt in zip(prompt_names, prompt_templates)
        }
        _default_chain = default_chain or ConversationChain(
            llm=ChatOpenAI(), output_key="text"
        )
        return cls(
            router_chain=router_chain,
            destination_chains=destination_chains,
            default_chain=_default_chain,
            **kwargs,
        )
