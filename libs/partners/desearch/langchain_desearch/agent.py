from datetime import datetime
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate
from .search_tools import search_tools  # Directly use the tools from search_tools


# Create the LangChain search agent
def create_search_agent(llm, **kwargs):
    """
    Creates a LangChain agent with the provided LLM and tools.

    Args:
        llm: Pre-initialized language model instance.
        **kwargs: Additional parameters for agent configuration.

    Returns:
        AgentExecutor: A fully-configured agent executor with tools and prompt.
    """
    # Define the role and system prompt
    current_datetime = datetime.now().isoformat()
    role_prompt = (
        f"You are a helpful assistant. Current date and time: {current_datetime}. "
        "You can use tools to search for information if needed."
    )

    # Create the prompt using ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", role_prompt),
            ("human", "{input_message}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # Create the agent using create_tool_calling_agent
    agent = create_tool_calling_agent(
        llm=llm,
        tools=search_tools,  # Use tools directly from search_tools
        prompt=prompt,
        **kwargs,
    )
    # Set the agent's name

    agent_executor = AgentExecutor(agent=agent, tools=search_tools, verbose=True)

    return agent_executor
