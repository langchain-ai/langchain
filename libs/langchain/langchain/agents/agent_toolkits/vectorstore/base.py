"""VectorStore agent."""

from typing import Any, Optional

from langchain_core._api import deprecated
from langchain_core.callbacks.base import BaseCallbackManager
from langchain_core.language_models import BaseLanguageModel

from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits.vectorstore.prompt import PREFIX, ROUTER_PREFIX
from langchain.agents.agent_toolkits.vectorstore.toolkit import (
    VectorStoreRouterToolkit,
    VectorStoreToolkit,
)
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.chains.llm import LLMChain


@deprecated(
    since="0.2.13",
    removal="1.0",
    message=(
        "This function will continue to be supported, but it is recommended for new "
        "use cases to be built with LangGraph. LangGraph offers a more flexible and "
        "full-featured framework for building agents, including support for "
        "tool-calling, persistence of state, and human-in-the-loop workflows. "
        "See API reference for this function for a replacement implementation: "
        "https://api.python.langchain.com/en/latest/agents/langchain.agents.agent_toolkits.vectorstore.base.create_vectorstore_agent.html "  # noqa: E501
        "Read more here on how to create agents that query vector stores: "
        "https://python.langchain.com/docs/how_to/qa_chat_history_how_to/#agents"
    ),
)
def create_vectorstore_agent(
    llm: BaseLanguageModel,
    toolkit: VectorStoreToolkit,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: str = PREFIX,
    verbose: bool = False,
    agent_executor_kwargs: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> AgentExecutor:
    """Construct a VectorStore agent from an LLM and tools.

    Note: this class is deprecated. See below for a replacement that uses tool
    calling methods and LangGraph. Install LangGraph with:

        .. code-block:: bash

            pip install -U langgraph

        .. code-block:: python

            from langchain_core.tools import create_retriever_tool
            from langchain_core.vectorstores import InMemoryVectorStore
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            from langgraph.prebuilt import create_react_agent

            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

            vector_store = InMemoryVectorStore.from_texts(
                [
                    "Dogs are great companions, known for their loyalty and friendliness.",
                    "Cats are independent pets that often enjoy their own space.",
                ],
                OpenAIEmbeddings(),
            )

            tool = create_retriever_tool(
                vector_store.as_retriever(),
                "pet_information_retriever",
                "Fetches information about pets.",
            )

            agent = create_react_agent(llm, [tool])

            for step in agent.stream(
                {"messages": [("human", "What are dogs known for?")]},
                stream_mode="values",
            ):
                step["messages"][-1].pretty_print()

    Args:
        llm (BaseLanguageModel): LLM that will be used by the agent
        toolkit (VectorStoreToolkit): Set of tools for the agent
        callback_manager (Optional[BaseCallbackManager], optional): Object to handle the callback [ Defaults to None. ]
        prefix (str, optional): The prefix prompt for the agent. If not provided uses default PREFIX.
        verbose (bool, optional): If you want to see the content of the scratchpad. [ Defaults to False ]
        agent_executor_kwargs (Optional[Dict[str, Any]], optional): If there is any other parameter you want to send to the agent. [ Defaults to None ]
        kwargs: Additional named parameters to pass to the ZeroShotAgent.

    Returns:
        AgentExecutor: Returns a callable AgentExecutor object. Either you can call it or use run method with the query to get the response
    """  # noqa: E501
    tools = toolkit.get_tools()
    prompt = ZeroShotAgent.create_prompt(tools, prefix=prefix)
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        callback_manager=callback_manager,
    )
    tool_names = [tool.name for tool in tools]
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names, **kwargs)
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        callback_manager=callback_manager,
        verbose=verbose,
        **(agent_executor_kwargs or {}),
    )


@deprecated(
    since="0.2.13",
    removal="1.0",
    message=(
        "This function will continue to be supported, but it is recommended for new "
        "use cases to be built with LangGraph. LangGraph offers a more flexible and "
        "full-featured framework for building agents, including support for "
        "tool-calling, persistence of state, and human-in-the-loop workflows. "
        "See API reference for this function for a replacement implementation: "
        "https://api.python.langchain.com/en/latest/agents/langchain.agents.agent_toolkits.vectorstore.base.create_vectorstore_router_agent.html "  # noqa: E501
        "Read more here on how to create agents that query vector stores: "
        "https://python.langchain.com/docs/how_to/qa_chat_history_how_to/#agents"
    ),
)
def create_vectorstore_router_agent(
    llm: BaseLanguageModel,
    toolkit: VectorStoreRouterToolkit,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: str = ROUTER_PREFIX,
    verbose: bool = False,
    agent_executor_kwargs: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> AgentExecutor:
    """Construct a VectorStore router agent from an LLM and tools.

    Note: this class is deprecated. See below for a replacement that uses tool
    calling methods and LangGraph. Install LangGraph with:

        .. code-block:: bash

            pip install -U langgraph

        .. code-block:: python

            from langchain_core.tools import create_retriever_tool
            from langchain_core.vectorstores import InMemoryVectorStore
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            from langgraph.prebuilt import create_react_agent

            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

            pet_vector_store = InMemoryVectorStore.from_texts(
                [
                    "Dogs are great companions, known for their loyalty and friendliness.",
                    "Cats are independent pets that often enjoy their own space.",
                ],
                OpenAIEmbeddings(),
            )

            food_vector_store = InMemoryVectorStore.from_texts(
                [
                    "Carrots are orange and delicious.",
                    "Apples are red and delicious.",
                ],
                OpenAIEmbeddings(),
            )

            tools = [
                create_retriever_tool(
                    pet_vector_store.as_retriever(),
                    "pet_information_retriever",
                    "Fetches information about pets.",
                ),
                create_retriever_tool(
                    food_vector_store.as_retriever(),
                    "food_information_retriever",
                    "Fetches information about food.",
                )
            ]

            agent = create_react_agent(llm, tools)

            for step in agent.stream(
                {"messages": [("human", "Tell me about carrots.")]},
                stream_mode="values",
            ):
                step["messages"][-1].pretty_print()

    Args:
        llm (BaseLanguageModel): LLM that will be used by the agent
        toolkit (VectorStoreRouterToolkit): Set of tools for the agent which have routing capability with multiple vector stores
        callback_manager (Optional[BaseCallbackManager], optional): Object to handle the callback [ Defaults to None. ]
        prefix (str, optional): The prefix prompt for the router agent. If not provided uses default ROUTER_PREFIX.
        verbose (bool, optional): If you want to see the content of the scratchpad. [ Defaults to False ]
        agent_executor_kwargs (Optional[Dict[str, Any]], optional): If there is any other parameter you want to send to the agent. [ Defaults to None ]
        kwargs: Additional named parameters to pass to the ZeroShotAgent.

    Returns:
        AgentExecutor: Returns a callable AgentExecutor object. Either you can call it or use run method with the query to get the response.
    """  # noqa: E501
    tools = toolkit.get_tools()
    prompt = ZeroShotAgent.create_prompt(tools, prefix=prefix)
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        callback_manager=callback_manager,
    )
    tool_names = [tool.name for tool in tools]
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names, **kwargs)
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        callback_manager=callback_manager,
        verbose=verbose,
        **(agent_executor_kwargs or {}),
    )
