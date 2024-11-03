"""Use a single chain to route an input to one of multiple llm chains."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core._api import deprecated
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate

from langchain.chains import ConversationChain
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.router.base import MultiRouteChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE


@deprecated(
    since="0.2.12",
    removal="1.0",
    message=(
        "Use RunnableLambda to select from multiple prompt templates. See example "
        "in API reference: "
        "https://api.python.langchain.com/en/latest/chains/langchain.chains.router.multi_prompt.MultiPromptChain.html"  # noqa: E501
    ),
)
class MultiPromptChain(MultiRouteChain):
    """A multi-route chain that uses an LLM router chain to choose amongst prompts.

    This class is deprecated. See below for a replacement, which offers several
    benefits, including streaming and batch support.

    Below is an example implementation:

        .. code-block:: python

            from operator import itemgetter
            from typing import Literal
            from typing_extensions import TypedDict

            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.runnables import RunnableLambda, RunnablePassthrough
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(model="gpt-4o-mini")

            prompt_1 = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are an expert on animals."),
                    ("human", "{query}"),
                ]
            )
            prompt_2 = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are an expert on vegetables."),
                    ("human", "{query}"),
                ]
            )

            chain_1 = prompt_1 | llm | StrOutputParser()
            chain_2 = prompt_2 | llm | StrOutputParser()

            route_system = "Route the user's query to either the animal or vegetable expert."
            route_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", route_system),
                    ("human", "{query}"),
                ]
            )


            class RouteQuery(TypedDict):
                \"\"\"Route query to destination.\"\"\"
                destination: Literal["animal", "vegetable"]


            route_chain = (
                route_prompt
                | llm.with_structured_output(RouteQuery)
                | itemgetter("destination")
            )

            chain = {
                "destination": route_chain,  # "animal" or "vegetable"
                "query": lambda x: x["query"],  # pass through input query
            } | RunnableLambda(
                # if animal, chain_1. otherwise, chain_2.
                lambda x: chain_1 if x["destination"] == "animal" else chain_2,
            )

            chain.invoke({"query": "what color are carrots"})
    """  # noqa: E501

    @property
    def output_keys(self) -> List[str]:
        return ["text"]

    @classmethod
    def from_prompts(
        cls,
        llm: BaseLanguageModel,
        prompt_infos: List[Dict[str, str]],
        default_chain: Optional[Chain] = None,
        **kwargs: Any,
    ) -> MultiPromptChain:
        """Convenience constructor for instantiating from destination prompts."""
        destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
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
        destination_chains = {}
        for p_info in prompt_infos:
            name = p_info["name"]
            prompt_template = p_info["prompt_template"]
            prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
            chain = LLMChain(llm=llm, prompt=prompt)
            destination_chains[name] = chain
        _default_chain = default_chain or ConversationChain(llm=llm, output_key="text")
        return cls(
            router_chain=router_chain,
            destination_chains=destination_chains,
            default_chain=_default_chain,
            **kwargs,
        )
