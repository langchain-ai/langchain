"""Use a single chain to route an input to one of multiple llm chains."""

from __future__ import annotations

from typing import Any

from langchain_core._api import deprecated
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from typing_extensions import override

from langchain_classic.chains import ConversationChain
from langchain_classic.chains.base import Chain
from langchain_classic.chains.llm import LLMChain
from langchain_classic.chains.router.base import MultiRouteChain
from langchain_classic.chains.router.llm_router import (
    LLMRouterChain,
    RouterOutputParser,
)
from langchain_classic.chains.router.multi_prompt_prompt import (
    MULTI_PROMPT_ROUTER_TEMPLATE,
)


@deprecated(
    since="0.2.12",
    removal="1.0",
    message=(
        "Please see migration guide here for recommended implementation: "
        "https://python.langchain.com/docs/versions/migrating_chains/multi_prompt_chain/"
    ),
)
class MultiPromptChain(MultiRouteChain):
    """A multi-route chain that uses an LLM router chain to choose amongst prompts.

    This class is deprecated. See below for a replacement, which offers several
    benefits, including streaming and batch support.

    Below is an example implementation:

        ```python
        from operator import itemgetter
        from typing import Literal

        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnableConfig
        from langchain_openai import ChatOpenAI
        from langgraph.graph import END, START, StateGraph
        from typing_extensions import TypedDict

        model = ChatOpenAI(model="gpt-4o-mini")

        # Define the prompts we will route to
        prompt_1 = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an expert on animals."),
                ("human", "{input}"),
            ]
        )
        prompt_2 = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an expert on vegetables."),
                ("human", "{input}"),
            ]
        )

        # Construct the chains we will route to. These format the input query
        # into the respective prompt, run it through a chat model, and cast
        # the result to a string.
        chain_1 = prompt_1 | model | StrOutputParser()
        chain_2 = prompt_2 | model | StrOutputParser()


        # Next: define the chain that selects which branch to route to.
        # Here we will take advantage of tool-calling features to force
        # the output to select one of two desired branches.
        route_system = "Route the user's query to either the animal "
        "or vegetable expert."
        route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", route_system),
                ("human", "{input}"),
            ]
        )


        # Define schema for output:
        class RouteQuery(TypedDict):
            \"\"\"Route query to destination expert.\"\"\"

            destination: Literal["animal", "vegetable"]


        route_chain = route_prompt | model.with_structured_output(RouteQuery)


        # For LangGraph, we will define the state of the graph to hold the query,
        # destination, and final answer.
        class State(TypedDict):
            query: str
            destination: RouteQuery
            answer: str


        # We define functions for each node, including routing the query:
        async def route_query(state: State, config: RunnableConfig):
            destination = await route_chain.ainvoke(state["query"], config)
            return {"destination": destination}


        # And one node for each prompt
        async def prompt_1(state: State, config: RunnableConfig):
            return {"answer": await chain_1.ainvoke(state["query"], config)}


        async def prompt_2(state: State, config: RunnableConfig):
            return {"answer": await chain_2.ainvoke(state["query"], config)}


        # We then define logic that selects the prompt based on the classification
        def select_node(state: State) -> Literal["prompt_1", "prompt_2"]:
            if state["destination"] == "animal":
                return "prompt_1"
            else:
                return "prompt_2"


        # Finally, assemble the multi-prompt chain. This is a sequence of two steps:
        # 1) Select "animal" or "vegetable" via the route_chain, and collect the
        # answer alongside the input query.
        # 2) Route the input query to chain_1 or chain_2, based on the
        # selection.
        graph = StateGraph(State)
        graph.add_node("route_query", route_query)
        graph.add_node("prompt_1", prompt_1)
        graph.add_node("prompt_2", prompt_2)

        graph.add_edge(START, "route_query")
        graph.add_conditional_edges("route_query", select_node)
        graph.add_edge("prompt_1", END)
        graph.add_edge("prompt_2", END)
        app = graph.compile()

        result = await app.ainvoke({"query": "what color are carrots"})
        print(result["destination"])
        print(result["answer"])

        ```
    """

    @property
    @override
    def output_keys(self) -> list[str]:
        return ["text"]

    @classmethod
    def from_prompts(
        cls,
        llm: BaseLanguageModel,
        prompt_infos: list[dict[str, str]],
        default_chain: Chain | None = None,
        **kwargs: Any,
    ) -> MultiPromptChain:
        """Convenience constructor for instantiating from destination prompts."""
        destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
        destinations_str = "\n".join(destinations)
        router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
            destinations=destinations_str,
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
