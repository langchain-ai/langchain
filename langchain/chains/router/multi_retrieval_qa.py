"""Use a single chain to route an input to one of multiple retrieval qa chains."""
from __future__ import annotations

from typing import Any, List, Mapping, Optional

from langchain.base_language import BaseLanguageModel
from langchain.chains import ConversationChain
from langchain.chains.base import Chain
from langchain.chains.conversation.prompt import DEFAULT_TEMPLATE
from langchain.chains.retrieval_qa.base import BaseRetrievalQA, RetrievalQA
from langchain.chains.router.base import MultiRouteChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_retrieval_prompt import (
    MULTI_RETRIEVAL_ROUTER_TEMPLATE,
)
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever


class MultiRetrievalQAChain(MultiRouteChain):
    """A multi-route chain that uses an LLM router chain to choose amongst retrieval
    qa chains."""

    router_chain: LLMRouterChain
    """Chain for deciding a destination chain and the input to it."""
    destination_chains: Mapping[str, BaseRetrievalQA]
    """Map of name to candidate chains that inputs can be routed to."""
    default_chain: Chain
    """Default chain to use when router doesn't map input to one of the destinations."""

    @property
    def output_keys(self) -> List[str]:
        return ["result"]

    @classmethod
    def from_retrievers(
        cls,
        llm: BaseLanguageModel,
        retriever_names: List[str],
        retriever_descriptions: List[str],
        retrievers: List[BaseRetriever],
        retriever_prompts: Optional[List[PromptTemplate]] = None,
        default_retriever: Optional[BaseRetriever] = None,
        default_prompt: Optional[PromptTemplate] = None,
        default_chain: Optional[Chain] = None,
        **kwargs: Any,
    ) -> MultiRetrievalQAChain:
        if default_prompt and not default_retriever:
            raise ValueError(
                "`default_retriever` must be specified if `default_prompt` is "
                "provided. Received only `default_prompt`."
            )
        destinations = [
            f"{name}: {description}"
            for name, description in zip(retriever_names, retriever_descriptions)
        ]
        destinations_str = "\n".join(destinations)
        router_template = MULTI_RETRIEVAL_ROUTER_TEMPLATE.format(
            destinations=destinations_str
        )
        router_prompt = PromptTemplate(
            template=router_template,
            input_variables=["input"],
            output_parser=RouterOutputParser(next_inputs_inner_key="query"),
        )
        router_chain = LLMRouterChain.from_llm(llm, router_prompt)
        destination_chains = {}
        for i, retriever in enumerate(retrievers):
            name = retriever_names[i]
            prompt = retriever_prompts[i] if retriever_prompts else None
            chain = RetrievalQA.from_llm(llm, prompt=prompt, retriever=retriever)
            destination_chains[name] = chain
        if default_chain:
            _default_chain = default_chain
        elif default_retriever:
            _default_chain = RetrievalQA.from_llm(
                llm, prompt=default_prompt, retriever=default_retriever
            )
        else:
            prompt_template = DEFAULT_TEMPLATE.replace("input", "query")
            prompt = PromptTemplate(
                template=prompt_template, input_variables=["history", "query"]
            )
            _default_chain = ConversationChain(
                llm=ChatOpenAI(), prompt=prompt, input_key="query", output_key="result"
            )
        return cls(
            router_chain=router_chain,
            destination_chains=destination_chains,
            default_chain=_default_chain,
            **kwargs,
        )
