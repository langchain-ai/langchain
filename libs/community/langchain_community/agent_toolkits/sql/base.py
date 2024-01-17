"""SQL agent."""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence, Union

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

from langchain_community.agent_toolkits.sql.prompt import (
    SQL_FUNCTIONS_SUFFIX,
    SQL_PREFIX,
    SQL_SUFFIX,
)
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

if TYPE_CHECKING:
    from langchain.agents.agent import AgentExecutor
    from langchain.agents.agent_types import AgentType
    from langchain_core.callbacks import BaseCallbackManager
    from langchain_core.language_models import BaseLanguageModel
    from langchain_core.tools import BaseTool

    from langchain_community.utilities.sql_database import SQLDatabase


def create_sql_agent(
    llm: BaseLanguageModel,
    toolkit: Optional[SQLDatabaseToolkit] = None,
    agent_type: Optional[Union[AgentType, Literal["openai-tools"]]] = None,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: str = SQL_PREFIX,
    suffix: Optional[str] = None,
    format_instructions: Optional[str] = None,
    input_variables: Optional[List[str]] = None,
    top_k: int = 10,
    max_iterations: Optional[int] = 15,
    max_execution_time: Optional[float] = None,
    early_stopping_method: str = "force",
    verbose: bool = False,
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    extra_tools: Sequence[BaseTool] = (),
    *,
    db: Optional[SQLDatabase] = None,
    **kwargs: Any,
) -> AgentExecutor:
    """Construct a SQL agent from an LLM and toolkit or database.

    Args:
        llm: Language model to use for the agent.
        toolkit: SQLDatabaseToolkit for the agent to use. Must provide exactly one of
            'toolkit' or 'db'. Specify 'toolkit' if you want to use a different model
            for the agent and the toolkit.
        agent_type: One of "openai-tools", "openai-functions", or
            "zero-shot-react-description". Defaults to "zero-shot-react-description".
            "openai-tools" is recommended over "openai-functions".
        callback_manager: DEPRECATED. Pass "callbacks" key into 'agent_executor_kwargs'
            instead to pass constructor callbacks to AgentExecutor.
        prefix: Prompt prefix string. Should contain variables "top_k" and "dialect".
        suffix: Prompt suffix string. Default depends on agent type.
        format_instructions: Formatting instructions to pass to
            ZeroShotAgent.create_prompt() when 'agent_type' is
            "zero-shot-react-description". Otherwise ignored.
        input_variables: DEPRECATED. Input variables to explicitly specify as part of
            ZeroShotAgent.create_prompt() when 'agent_type' is
            "zero-shot-react-description". Otherwise ignored.
        top_k: Number of rows to query for by default.
        max_iterations: Passed to AgentExecutor init.
        max_execution_time: Passed to AgentExecutor init.
        early_stopping_method: Passed to AgentExecutor init.
        verbose: AgentExecutor verbosity.
        agent_executor_kwargs: Arbitrary additional AgentExecutor args.
        extra_tools: Additional tools to give to agent on top of the ones that come with
            SQLDatabaseToolkit.
        db: SQLDatabase from which to create a SQLDatabaseToolkit. Toolkit is created
            using 'db' and 'llm'. Must provide exactly one of 'db' or 'toolkit'.
        **kwargs: DEPRECATED. Not used, kept for backwards compatibility.

    Returns:
        An AgentExecutor with the specified agent_type agent.

    Example:

        .. code-block:: python

        from langchain_openai import ChatOpenAI
        from langchain_community.agent_toolkits import create_sql_agent
        from langchain_community.utilities import SQLDatabase

        db = SQLDatabase.from_uri("sqlite:///Chinook.db")
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

    """  # noqa: E501
    from langchain.agents import (
        create_openai_functions_agent,
        create_openai_tools_agent,
        create_react_agent,
    )
    from langchain.agents.agent import (
        AgentExecutor,
        RunnableAgent,
        RunnableMultiActionAgent,
    )
    from langchain.agents.agent_types import AgentType
    from langchain.agents.mrkl.base import ZeroShotAgent

    if toolkit is None and db is None:
        raise ValueError(
            "Must provide exactly one of 'toolkit' or 'db'. Received neither."
        )
    if toolkit and db:
        raise ValueError(
            "Must provide exactly one of 'toolkit' or 'db'. Received both."
        )
    if kwargs:
        warnings.warn(
            f"Received additional kwargs {kwargs} which are no longer supported."
        )

    toolkit = toolkit or SQLDatabaseToolkit(llm=llm, db=db)
    agent_type = agent_type or AgentType.ZERO_SHOT_REACT_DESCRIPTION
    tools = toolkit.get_tools() + list(extra_tools)
    prefix = prefix.format(dialect=toolkit.dialect, top_k=top_k)

    if agent_type == AgentType.ZERO_SHOT_REACT_DESCRIPTION:
        prompt_params = (
            {"format_instructions": format_instructions}
            if format_instructions is not None
            else {}
        )
        prompt = ZeroShotAgent.create_prompt(
            tools,
            prefix=prefix,
            suffix=suffix or SQL_SUFFIX,
            input_variables=input_variables,
            **prompt_params,
        )
        agent = RunnableAgent(
            runnable=create_react_agent(llm, tools, prompt), input_keys_arg=["input"]
        )

    elif agent_type == AgentType.OPENAI_FUNCTIONS:
        messages = [
            SystemMessage(content=prefix),
            HumanMessagePromptTemplate.from_template("{input}"),
            AIMessage(content=suffix or SQL_FUNCTIONS_SUFFIX),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
        prompt = ChatPromptTemplate.from_messages(messages)
        agent = RunnableAgent(
            runnable=create_openai_functions_agent(llm, tools, prompt),
            input_keys_arg=["input"],
        )
    elif agent_type == "openai-tools":
        messages = [
            SystemMessage(content=prefix),
            HumanMessagePromptTemplate.from_template("{input}"),
            AIMessage(content=suffix or SQL_FUNCTIONS_SUFFIX),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
        prompt = ChatPromptTemplate.from_messages(messages)
        agent = RunnableMultiActionAgent(
            runnable=create_openai_tools_agent(llm, tools, prompt),
            input_keys_arg=["input"],
        )

    else:
        raise ValueError(f"Agent type {agent_type} not supported at the moment.")

    return AgentExecutor(
        agent=agent,
        tools=tools,
        callback_manager=callback_manager,
        verbose=verbose,
        max_iterations=max_iterations,
        max_execution_time=max_execution_time,
        early_stopping_method=early_stopping_method,
        **(agent_executor_kwargs or {}),
    )
