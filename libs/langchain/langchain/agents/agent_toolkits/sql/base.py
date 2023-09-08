"""SQL agent."""
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from langchain.agents.agent import AgentExecutor, BaseSingleActionAgent
from langchain.agents.agent_toolkits.sql.prompt import (
    SQL_FUNCTIONS_SUFFIX,
    SQL_PREFIX,
    SQL_SUFFIX,
)
from langchain.agents.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
from langchain.agents.openai_functions_agent.base import (
    OpenAIFunctionsAgent,
    _format_intermediate_steps,
    _parse_ai_message,
)
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.manager import Callbacks
from langchain.chains.llm import LLMChain
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import AIMessage, SystemMessage

from langchain.schema import AgentAction, AgentFinish, BasePromptTemplate
from langchain.tools import BaseTool


class SQLOpenAIFunctionsAgent(OpenAIFunctionsAgent):
    llm: BaseLanguageModel
    tools: Sequence[BaseTool]
    prompt: BasePromptTemplate
    automatic_retrievers: Sequence[tuple[str, Callable]]

    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        with_functions: bool = True,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date, along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        agent_scratchpad = _format_intermediate_steps(intermediate_steps)
        selected_inputs = {
            k: kwargs[k]
            for k in self.prompt.input_variables
            if k != "agent_scratchpad" and k != "retrieved_results"
        }

        # Add the automatic retrievers' results to the AI message
        retrieved_results = ""
        for intro_message, retriever in self.automatic_retrievers:
            result = retriever(selected_inputs["input"])
            retrieved_results += intro_message + "\n" + str(result)
            retrieved_results += "\n\n"

        full_inputs = dict(
            **selected_inputs,
            agent_scratchpad=agent_scratchpad,
            retrieved_results=retrieved_results,
        )

        prompt = self.prompt.format_prompt(**full_inputs)

        messages = prompt.to_messages()

        print(messages)

        if with_functions:
            predicted_message = self.llm.predict_messages(
                messages,
                functions=self.functions,
                callbacks=callbacks,
            )
        else:
            predicted_message = self.llm.predict_messages(
                messages,
                callbacks=callbacks,
            )
        agent_decision = _parse_ai_message(predicted_message)
        return agent_decision


def create_sql_agent(
    llm: BaseLanguageModel,
    toolkit: SQLDatabaseToolkit,
    agent_type: AgentType = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: str = SQL_PREFIX,
    suffix: Optional[str] = None,
    format_instructions: str = FORMAT_INSTRUCTIONS,
    input_variables: Optional[List[str]] = None,
    top_k: int = 10,
    max_iterations: Optional[int] = 15,
    max_execution_time: Optional[float] = None,
    early_stopping_method: str = "force",
    verbose: bool = False,
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    extra_tools: Sequence[BaseTool] = (),
    automatic_retrievers: Sequence[tuple[str, Callable]] = (),
    **kwargs: Dict[str, Any],
) -> AgentExecutor:
    """Construct an SQL agent from an LLM and tools."""
    tools = toolkit.get_tools() + list(extra_tools)
    prefix = prefix.format(dialect=toolkit.dialect, top_k=top_k)
    agent: BaseSingleActionAgent

    prefix += "\n These are the tables in the database: "
    prefix += ", ".join(toolkit.db.get_usable_table_names())

    print(prefix)

    if agent_type == AgentType.ZERO_SHOT_REACT_DESCRIPTION:
        prompt = ZeroShotAgent.create_prompt(
            tools,
            prefix=prefix,
            suffix=suffix or SQL_SUFFIX,
            format_instructions=format_instructions,
            input_variables=input_variables,
        )

        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names, **kwargs)

    elif agent_type == AgentType.OPENAI_FUNCTIONS:
        messages = [
            SystemMessage(content=prefix),
            HumanMessagePromptTemplate.from_template("{input}"),
            AIMessage(content=suffix or SQL_FUNCTIONS_SUFFIX),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
        input_variables = ["input", "agent_scratchpad"]

        if automatic_retrievers:
            messages.insert(
                1, AIMessagePromptTemplate.from_template("{retrieved_results}")
            )
            input_variables.append("retrieved_results")

        _prompt = ChatPromptTemplate(input_variables=input_variables, messages=messages)

        agent = SQLOpenAIFunctionsAgent(
            llm=llm,
            prompt=_prompt,
            tools=tools,
            callback_manager=callback_manager,
            automatic_retrievers=automatic_retrievers,
            **kwargs,
        )
    else:
        raise ValueError(f"Agent type {agent_type} not supported at the moment.")

    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        callback_manager=callback_manager,
        verbose=verbose,
        max_iterations=max_iterations,
        max_execution_time=max_execution_time,
        early_stopping_method=early_stopping_method,
        **(agent_executor_kwargs or {}),
    )
