import re
from typing import Any, List, Optional, Sequence, Tuple, Union

from langchain_core._api import deprecated
from langchain_core.agents import AgentAction
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.pydantic_v1 import Field
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool

from langchain.agents.agent import Agent, AgentOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.agents.structured_chat.output_parser import (
    StructuredChatOutputParserWithRetries,
)
from langchain.agents.structured_chat.prompt import FORMAT_INSTRUCTIONS, PREFIX, SUFFIX
from langchain.chains.llm import LLMChain
from langchain.tools.render import ToolsRenderer, render_text_description_and_args

HUMAN_MESSAGE_TEMPLATE = "{input}\n\n{agent_scratchpad}"


@deprecated("0.1.0", alternative="create_structured_chat_agent", removal="0.2.0")
class StructuredChatAgent(Agent):
    """Structured Chat Agent."""

    output_parser: AgentOutputParser = Field(
        default_factory=StructuredChatOutputParserWithRetries
    )
    """Output parser for the agent."""

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "Observation: "

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the llm call with."""
        return "Thought:"

    def _construct_scratchpad(
        self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> str:
        agent_scratchpad = super()._construct_scratchpad(intermediate_steps)
        if not isinstance(agent_scratchpad, str):
            raise ValueError("agent_scratchpad should be of type string.")
        if agent_scratchpad:
            return (
                f"This was your previous work "
                f"(but I haven't seen any of it! I only see what "
                f"you return as final answer):\n{agent_scratchpad}"
            )
        else:
            return agent_scratchpad

    @classmethod
    def _validate_tools(cls, tools: Sequence[BaseTool]) -> None:
        pass

    @classmethod
    def _get_default_output_parser(
        cls, llm: Optional[BaseLanguageModel] = None, **kwargs: Any
    ) -> AgentOutputParser:
        return StructuredChatOutputParserWithRetries.from_llm(llm=llm)

    @property
    def _stop(self) -> List[str]:
        return ["Observation:"]

    @classmethod
    def create_prompt(
        cls,
        tools: Sequence[BaseTool],
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        human_message_template: str = HUMAN_MESSAGE_TEMPLATE,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        input_variables: Optional[List[str]] = None,
        memory_prompts: Optional[List[BasePromptTemplate]] = None,
    ) -> BasePromptTemplate:
        tool_strings = []
        for tool in tools:
            args_schema = re.sub("}", "}}", re.sub("{", "{{", str(tool.args)))
            tool_strings.append(f"{tool.name}: {tool.description}, args: {args_schema}")
        formatted_tools = "\n".join(tool_strings)
        tool_names = ", ".join([tool.name for tool in tools])
        format_instructions = format_instructions.format(tool_names=tool_names)
        template = "\n\n".join([prefix, formatted_tools, format_instructions, suffix])
        if input_variables is None:
            input_variables = ["input", "agent_scratchpad"]
        _memory_prompts = memory_prompts or []
        messages = [
            SystemMessagePromptTemplate.from_template(template),
            *_memory_prompts,
            HumanMessagePromptTemplate.from_template(human_message_template),
        ]
        return ChatPromptTemplate(input_variables=input_variables, messages=messages)  # type: ignore[arg-type]

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        output_parser: Optional[AgentOutputParser] = None,
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        human_message_template: str = HUMAN_MESSAGE_TEMPLATE,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        input_variables: Optional[List[str]] = None,
        memory_prompts: Optional[List[BasePromptTemplate]] = None,
        **kwargs: Any,
    ) -> Agent:
        """Construct an agent from an LLM and tools."""
        cls._validate_tools(tools)
        prompt = cls.create_prompt(
            tools,
            prefix=prefix,
            suffix=suffix,
            human_message_template=human_message_template,
            format_instructions=format_instructions,
            input_variables=input_variables,
            memory_prompts=memory_prompts,
        )
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        _output_parser = output_parser or cls._get_default_output_parser(llm=llm)
        return cls(
            llm_chain=llm_chain,
            allowed_tools=tool_names,
            output_parser=_output_parser,
            **kwargs,
        )

    @property
    def _agent_type(self) -> str:
        raise ValueError


def create_structured_chat_agent(
    llm: BaseLanguageModel,
    tools: Sequence[BaseTool],
    prompt: ChatPromptTemplate,
    tools_renderer: ToolsRenderer = render_text_description_and_args,
    *,
    stop_sequence: Union[bool, List[str]] = True,
) -> Runnable:
    """Create an agent aimed at supporting tools with multiple inputs.

    Args:
        llm: LLM to use as the agent.
        tools: Tools this agent has access to.
        prompt: The prompt to use. See Prompt section below for more.
        stop_sequence: bool or list of str.
            If True, adds a stop token of "Observation:" to avoid hallucinates.
            If False, does not add a stop token.
            If a list of str, uses the provided list as the stop tokens.

            Default is True. You may to set this to False if the LLM you are using
            does not support stop sequences.
        tools_renderer: This controls how the tools are converted into a string and
            then passed into the LLM. Default is `render_text_description`.

    Returns:
        A Runnable sequence representing an agent. It takes as input all the same input
        variables as the prompt passed in does. It returns as output either an
        AgentAction or AgentFinish.

    Examples:

        .. code-block:: python

            from langchain import hub
            from langchain_community.chat_models import ChatOpenAI
            from langchain.agents import AgentExecutor, create_structured_chat_agent

            prompt = hub.pull("hwchase17/structured-chat-agent")
            model = ChatOpenAI()
            tools = ...

            agent = create_structured_chat_agent(model, tools, prompt)
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

        The prompt must have input keys:
            * `tools`: contains descriptions and arguments for each tool.
            * `tool_names`: contains all tool names.
            * `agent_scratchpad`: contains previous agent actions and tool outputs as a string.

        Here's an example:

        .. code-block:: python

            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

            system = '''Respond to the human as helpfully and accurately as possible. You have access to the following tools:

            {tools}

            Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

            Valid "action" values: "Final Answer" or {tool_names}

            Provide only ONE action per $JSON_BLOB, as shown:

            ```
            {{
              "action": $TOOL_NAME,
              "action_input": $INPUT
            }}
            ```

            Follow this format:

            Question: input question to answer
            Thought: consider previous and subsequent steps
            Action:
            ```
            $JSON_BLOB
            ```
            Observation: action result
            ... (repeat Thought/Action/Observation N times)
            Thought: I know what to respond
            Action:
            ```
            {{
              "action": "Final Answer",
              "action_input": "Final response to human"
            }}

            Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation'''

            human = '''{input}

            {agent_scratchpad}

            (reminder to respond in a JSON blob no matter what)'''

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system),
                    MessagesPlaceholder("chat_history", optional=True),
                    ("human", human),
                ]
            )
    """  # noqa: E501
    missing_vars = {"tools", "tool_names", "agent_scratchpad"}.difference(
        prompt.input_variables + list(prompt.partial_variables)
    )
    if missing_vars:
        raise ValueError(f"Prompt missing required variables: {missing_vars}")

    prompt = prompt.partial(
        tools=tools_renderer(list(tools)),
        tool_names=", ".join([t.name for t in tools]),
    )
    if stop_sequence:
        stop = ["\nObservation"] if stop_sequence is True else stop_sequence
        llm_with_stop = llm.bind(stop=stop)
    else:
        llm_with_stop = llm

    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
        )
        | prompt
        | llm_with_stop
        | JSONAgentOutputParser()
    )
    return agent
