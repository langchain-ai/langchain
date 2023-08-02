from typing import Optional, List
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.tools.base import Tool, BaseTool
from langchain.schema.retriever import BaseRetriever
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.memory.token_buffer import ConversationTokenBufferMemory
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import SystemMessage
from langchain.prompts.chat import MessagesPlaceholder
from langchain.agents.agent import AgentExecutor

def create_retriever_tool(retriever: BaseRetriever, name: str, description: str) -> Tool:
    """Create a tool to do retrieval of documents.

    Args:
        retriever: The retriever to use for the retrieval
        name: The name for the tool. This will be passed to the language model,
            so should be unique and somewhat descriptive.
        description: The description for the tool. This will be passed to the language
            model, so should be descriptive.

    Returns:
        Tool class to pass to an agent
    """
    return Tool(name=name, description=description, func=retriever.get_relevant_documents)


def _get_default_system_message() -> SystemMessage:
    return SystemMessage(
        content=(
            "Do your best to answer the questions. "
            "Feel free to use any tools available to look up "
            "relevant information, only if neccessary"
        )
    )


def create_conversational_retrieval_agent(
        llm: BaseLanguageModel,
        tools: List[BaseTool],
        remember_intermediate_steps: bool = True,
        memory_key="chat_history",
        system_message: Optional[SystemMessage] = None,
        verbose: bool = False
):
    """A convenience method for creating a conversational retrieval agent.

    Args:
        llm: The language model to use, should usually be ChatOpenAI
        tools: A list of tools the agent has access to
        remember_intermediate_steps: Whether the agent should remember intermediate
            steps or not. Intermediate steps refer to prior action/observation
            pairs from previous questions. The benefit of remembering these is if
            there is relevant information in there, the agent can use it to answer
            follow up questions. The downside is it will take up more tokens.
        memory_key: The name of the memory key in the prompt.
        system_message: The system message to use. By default, a basic one will
            be used.
        verbose: Whether or not the final AgentExecutor should be verbose or not,
            defaults to False.

    Returns:
        An agent executor initialized appropriately
    """
    if remember_intermediate_steps:
        memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm)
    else:
        memory = ConversationTokenBufferMemory(
            memory_key=memory_key,
            return_messages=True,
            output_key="output",
            llm=llm,
            max_token_limit=12000
        )

    _system_message = system_message or _get_default_system_message()
    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=_system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)]
    )
    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=verbose,
                                   return_intermediate_steps=remember_intermediate_steps)


