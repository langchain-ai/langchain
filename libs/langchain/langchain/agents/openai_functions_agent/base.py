"""Module implements an agent that uses OpenAI's APIs function enabled API."""

from collections.abc import Sequence
from typing import Any, Optional, Union

from langchain_core._api import deprecated
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackManager, Callbacks
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
)
from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.prompts.message import BaseMessagePromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_function
from pydantic import model_validator
from typing_extensions import Self

from langchain.agents import BaseSingleActionAgent
from langchain.agents.format_scratchpad.openai_functions import (
    format_to_openai_function_messages,
)
from langchain.agents.output_parsers.openai_functions import (
    OpenAIFunctionsAgentOutputParser,
)


@deprecated("0.1.0", alternative="create_openai_functions_agent", removal="1.0")
class OpenAIFunctionsAgent(BaseSingleActionAgent):
    """An Agent driven by OpenAIs function powered API.

    Args:
        llm: This should be an instance of ChatOpenAI, specifically a model
            that supports using `functions`.
        tools: The tools this agent has access to.
        prompt: The prompt for this agent, should support agent_scratchpad as one
            of the variables. For an easy way to construct this prompt, use
            `OpenAIFunctionsAgent.create_prompt(...)`
        output_parser: The output parser for this agent. Should be an instance of
            OpenAIFunctionsAgentOutputParser.
            Defaults to OpenAIFunctionsAgentOutputParser.
    """

    llm: BaseLanguageModel
    tools: Sequence[BaseTool]
    prompt: BasePromptTemplate
    output_parser: type[OpenAIFunctionsAgentOutputParser] = (
        OpenAIFunctionsAgentOutputParser
    )

    def get_allowed_tools(self) -> list[str]:
        """Get allowed tools."""
        return [t.name for t in self.tools]

    @model_validator(mode="after")
    def validate_prompt(self) -> Self:
        """Validate prompt.

        Args:
            values: Values to validate.

        Returns:
            Validated values.

        Raises:
            ValueError: If `agent_scratchpad` is not in the prompt.
        """
        prompt: BasePromptTemplate = self.prompt
        if "agent_scratchpad" not in prompt.input_variables:
            raise ValueError(
                "`agent_scratchpad` should be one of the variables in the prompt, "
                f"got {prompt.input_variables}"
            )
        return self

    @property
    def input_keys(self) -> list[str]:
        """Get input keys. Input refers to user input here."""
        return ["input"]

    @property
    def functions(self) -> list[dict]:
        """Get functions."""

        return [dict(convert_to_openai_function(t)) for t in self.tools]

    def plan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        with_functions: bool = True,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations.
            callbacks: Callbacks to use. Defaults to None.
            with_functions: Whether to use functions. Defaults to True.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
            If the agent is finished, returns an AgentFinish.
            If the agent is not finished, returns an AgentAction.
        """
        agent_scratchpad = format_to_openai_function_messages(intermediate_steps)
        selected_inputs = {
            k: kwargs[k] for k in self.prompt.input_variables if k != "agent_scratchpad"
        }
        full_inputs = dict(**selected_inputs, agent_scratchpad=agent_scratchpad)
        prompt = self.prompt.format_prompt(**full_inputs)
        messages = prompt.to_messages()
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
        agent_decision = self.output_parser._parse_ai_message(predicted_message)
        return agent_decision

    async def aplan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """Async given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations.
            callbacks: Callbacks to use. Defaults to None.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
            If the agent is finished, returns an AgentFinish.
            If the agent is not finished, returns an AgentAction.
        """
        agent_scratchpad = format_to_openai_function_messages(intermediate_steps)
        selected_inputs = {
            k: kwargs[k] for k in self.prompt.input_variables if k != "agent_scratchpad"
        }
        full_inputs = dict(**selected_inputs, agent_scratchpad=agent_scratchpad)
        prompt = self.prompt.format_prompt(**full_inputs)
        messages = prompt.to_messages()
        predicted_message = await self.llm.apredict_messages(
            messages, functions=self.functions, callbacks=callbacks
        )
        agent_decision = self.output_parser._parse_ai_message(predicted_message)
        return agent_decision

    def return_stopped_response(
        self,
        early_stopping_method: str,
        intermediate_steps: list[tuple[AgentAction, str]],
        **kwargs: Any,
    ) -> AgentFinish:
        """Return response when agent has been stopped due to max iterations.

        Args:
            early_stopping_method: The early stopping method to use.
            intermediate_steps: Intermediate steps.
            **kwargs: User inputs.

        Returns:
            AgentFinish.

        Raises:
            ValueError: If `early_stopping_method` is not `force` or `generate`.
            ValueError: If `agent_decision` is not an AgentAction.
        """
        if early_stopping_method == "force":
            # `force` just returns a constant string
            return AgentFinish(
                {"output": "Agent stopped due to iteration limit or time limit."}, ""
            )
        elif early_stopping_method == "generate":
            # Generate does one final forward pass
            agent_decision = self.plan(
                intermediate_steps, with_functions=False, **kwargs
            )
            if isinstance(agent_decision, AgentFinish):
                return agent_decision
            else:
                raise ValueError(
                    f"got AgentAction with no functions provided: {agent_decision}"
                )
        else:
            raise ValueError(
                "early_stopping_method should be one of `force` or `generate`, "
                f"got {early_stopping_method}"
            )

    @classmethod
    def create_prompt(
        cls,
        system_message: Optional[SystemMessage] = SystemMessage(
            content="You are a helpful AI assistant."
        ),
        extra_prompt_messages: Optional[list[BaseMessagePromptTemplate]] = None,
    ) -> ChatPromptTemplate:
        """Create prompt for this agent.

        Args:
            system_message: Message to use as the system message that will be the
                first in the prompt.
            extra_prompt_messages: Prompt messages that will be placed between the
                system message and the new human input.

        Returns:
            A prompt template to pass into this agent.
        """
        _prompts = extra_prompt_messages or []
        messages: list[Union[BaseMessagePromptTemplate, BaseMessage]]
        if system_message:
            messages = [system_message]
        else:
            messages = []

        messages.extend(
            [
                *_prompts,
                HumanMessagePromptTemplate.from_template("{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        return ChatPromptTemplate(messages=messages)

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        extra_prompt_messages: Optional[list[BaseMessagePromptTemplate]] = None,
        system_message: Optional[SystemMessage] = SystemMessage(
            content="You are a helpful AI assistant."
        ),
        **kwargs: Any,
    ) -> BaseSingleActionAgent:
        """Construct an agent from an LLM and tools.

        Args:
            llm: The LLM to use as the agent.
            tools: The tools to use.
            callback_manager: The callback manager to use. Defaults to None.
            extra_prompt_messages: Extra prompt messages to use. Defaults to None.
            system_message: The system message to use.
                Defaults to a default system message.
            kwargs: Additional parameters to pass to the agent.
        """
        prompt = cls.create_prompt(
            extra_prompt_messages=extra_prompt_messages,
            system_message=system_message,
        )
        return cls(  # type: ignore[call-arg]
            llm=llm,
            prompt=prompt,
            tools=tools,
            callback_manager=callback_manager,
            **kwargs,
        )


def create_openai_functions_agent(
    llm: BaseLanguageModel, tools: Sequence[BaseTool], prompt: ChatPromptTemplate
) -> Runnable:
    """Create an agent that uses OpenAI function calling.

    Args:
        llm: LLM to use as the agent. Should work with OpenAI function calling,
            so either be an OpenAI model that supports that or a wrapper of
            a different model that adds in equivalent support.
        tools: Tools this agent has access to.
        prompt: The prompt to use. See Prompt section below for more.

    Returns:
        A Runnable sequence representing an agent. It takes as input all the same input
            variables as the prompt passed in does. It returns as output either an
            AgentAction or AgentFinish.

    Raises:
        ValueError: If `agent_scratchpad` is not in the prompt.

    Example:

        Creating an agent with no memory

        .. code-block:: python

            from langchain_community.chat_models import ChatOpenAI
            from langchain.agents import AgentExecutor, create_openai_functions_agent
            from langchain import hub

            prompt = hub.pull("hwchase17/openai-functions-agent")
            model = ChatOpenAI()
            tools = ...

            agent = create_openai_functions_agent(model, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools)

            agent_executor.invoke({"input": "hi"})

            # Using with chat history
            from langchain_core.messages import AIMessage, HumanMessage
            agent_executor.invoke(
                {
                    "input": "what's my name?",
                    "chat_history": [
                        HumanMessage(content="hi! my name is bob"),
                        AIMessage(content="Hello Bob! How can I assist you today?"),
                    ],
                }
            )

    Prompt:

        The agent prompt must have an `agent_scratchpad` key that is a
            ``MessagesPlaceholder``. Intermediate agent actions and tool output
            messages will be passed in here.

        Here's an example:

        .. code-block:: python

            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful assistant"),
                    MessagesPlaceholder("chat_history", optional=True),
                    ("human", "{input}"),
                    MessagesPlaceholder("agent_scratchpad"),
                ]
            )
    """
    if "agent_scratchpad" not in (
        prompt.input_variables + list(prompt.partial_variables)
    ):
        raise ValueError(
            "Prompt must have input variable `agent_scratchpad`, but wasn't found. "
            f"Found {prompt.input_variables} instead."
        )
    llm_with_tools = llm.bind(functions=[convert_to_openai_function(t) for t in tools])
    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_openai_function_messages(
                x["intermediate_steps"]
            )
        )
        | prompt
        | llm_with_tools
        | OpenAIFunctionsAgentOutputParser()
    )
    return agent
