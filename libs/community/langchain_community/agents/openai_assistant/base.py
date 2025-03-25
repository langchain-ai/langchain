from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Optional,
    Sequence,
    Type,
    Union,
)

from langchain.agents.openai_assistant.base import OpenAIAssistantRunnable, OutputType
from langchain_core._api import beta
from langchain_core.callbacks import CallbackManager
from langchain_core.load import dumpd
from langchain_core.runnables import RunnableConfig, ensure_config
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

if TYPE_CHECKING:
    import openai
    from openai._types import NotGiven
    from openai.types.beta.assistant import ToolResources as AssistantToolResources


def _get_openai_client() -> openai.OpenAI:
    """Get the OpenAI client.

    Returns:
        openai.OpenAI: OpenAI client

    Raises:
        ImportError: If `openai` is not installed.
        AttributeError: If the installed `openai` version is not compatible.
    """
    try:
        import openai

        return openai.OpenAI(default_headers={"OpenAI-Beta": "assistants=v2"})
    except ImportError as e:
        raise ImportError(
            "Unable to import openai, please install with `pip install openai`."
        ) from e
    except AttributeError as e:
        raise AttributeError(
            "Please make sure you are using a v1.23-compatible version of openai. You "
            'can install with `pip install "openai>=1.23"`.'
        ) from e


def _get_openai_async_client() -> openai.AsyncOpenAI:
    """Get the async OpenAI client.

    Returns:
        openai.AsyncOpenAI: Async OpenAI client

    Raises:
        ImportError: If `openai` is not installed.
        AttributeError: If the installed `openai` version is not compatible.
    """
    try:
        import openai

        return openai.AsyncOpenAI(default_headers={"OpenAI-Beta": "assistants=v2"})
    except ImportError as e:
        raise ImportError(
            "Unable to import openai, please install with `pip install openai`."
        ) from e
    except AttributeError as e:
        raise AttributeError(
            "Please make sure you are using a v1.23-compatible version of openai. You "
            'can install with `pip install "openai>=1.23"`.'
        ) from e


def _convert_file_ids_into_attachments(file_ids: list) -> list:
    """Convert file_ids into attachments
    File search and Code interpreter will be turned on by default.

    Args:
        file_ids (list): List of file_ids that need to be converted into attachments.

    Returns:
        list: List of attachments converted from file_ids.
    """
    attachments = []
    for id in file_ids:
        attachments.append(
            {
                "file_id": id,
                "tools": [{"type": "file_search"}, {"type": "code_interpreter"}],
            }
        )
    return attachments


def _is_assistants_builtin_tool(
    tool: Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool],
) -> bool:
    """Determine if tool corresponds to OpenAI Assistants built-in.

    Args:
        tool (Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]):
            Tool that needs to be determined.

    Returns:
        A boolean response of true or false indicating if the tool corresponds to
            OpenAI Assistants built-in.
    """
    assistants_builtin_tools = ("code_interpreter", "retrieval", "file_search")
    return (
        isinstance(tool, dict)
        and ("type" in tool)
        and (tool["type"] in assistants_builtin_tools)
    )


def _get_assistants_tool(
    tool: Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool],
) -> Dict[str, Any]:
    """Convert a raw function/class to an OpenAI tool.

    Note that OpenAI assistants supports several built-in tools,
    such as "code_interpreter" and "retrieval."

    Args:
        tool (Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]):
            Tools or functions that need to be converted to OpenAI tools.

    Returns:
        Dict[str, Any]: A dictionary of tools that are converted into OpenAI tools.
    """
    if _is_assistants_builtin_tool(tool):
        return tool  # type: ignore
    else:
        return convert_to_openai_tool(tool)


@beta()
class OpenAIAssistantV2Runnable(OpenAIAssistantRunnable):
    """Run an OpenAI Assistant.

    Attributes:
        client (Any): OpenAI or AzureOpenAI client.
        async_client (Any): Async OpenAI or AzureOpenAI client.
        assistant_id (str): OpenAI assistant ID.
        check_every_ms (float): Frequency to check progress in milliseconds.
        as_agent (bool): Whether to use the assistant as a LangChain agent.

    Example using OpenAI tools:
        .. code-block:: python

            from langchain.agents.openai_assistant import OpenAIAssistantV2Runnable

            assistant = OpenAIAssistantV2Runnable.create_assistant(
                name="math assistant",
                instructions="You are a personal math tutor. Write and run code to answer math questions.",
                tools=[{"type": "code_interpreter"}],
                model="gpt-4-1106-preview"
            )
            output = assistant.invoke({"content": "What's 10 - 4 raised to the 2.7"})

    Example using custom tools and AgentExecutor:
        .. code-block:: python

            from langchain.agents.openai_assistant import OpenAIAssistantV2Runnable
            from langchain.agents import AgentExecutor
            from langchain.tools import E2BDataAnalysisTool


            tools = [E2BDataAnalysisTool(api_key="...")]
            agent = OpenAIAssistantV2Runnable.create_assistant(
                name="langchain assistant e2b tool",
                instructions="You are a personal math tutor. Write and run code to answer math questions.",
                tools=tools,
                model="gpt-4-1106-preview",
                as_agent=True
            )

            agent_executor = AgentExecutor(agent=agent, tools=tools)
            agent_executor.invoke({"content": "Analyze the data..."})

    Example using custom tools and custom execution:
        .. code-block:: python

            from langchain.agents.openai_assistant import OpenAIAssistantV2Runnable
            from langchain.agents import AgentExecutor
            from langchain_core.agents import AgentFinish
            from langchain.tools import E2BDataAnalysisTool


            tools = [E2BDataAnalysisTool(api_key="...")]
            agent = OpenAIAssistantV2Runnable.create_assistant(
                name="langchain assistant e2b tool",
                instructions="You are a personal math tutor. Write and run code to answer math questions.",
                tools=tools,
                model="gpt-4-1106-preview",
                as_agent=True
            )

            def execute_agent(agent, tools, input):
                tool_map = {tool.name: tool for tool in tools}
                response = agent.invoke(input)
                while not isinstance(response, AgentFinish):
                    tool_outputs = []
                    for action in response:
                        tool_output = tool_map[action.tool].invoke(action.tool_input)
                        tool_outputs.append({"output": tool_output, "tool_call_id": action.tool_call_id})
                    response = agent.invoke(
                        {
                            "tool_outputs": tool_outputs,
                            "run_id": action.run_id,
                            "thread_id": action.thread_id
                        }
                    )

                return response

            response = execute_agent(agent, tools, {"content": "What's 10 - 4 raised to the 2.7"})
            next_response = execute_agent(agent, tools, {"content": "now add 17.241", "thread_id": response.thread_id})

    """  # noqa: E501

    client: Any = Field(default_factory=_get_openai_client)
    """OpenAI or AzureOpenAI client."""
    async_client: Any = None
    """OpenAI or AzureOpenAI async client."""
    assistant_id: str
    """OpenAI assistant id."""
    check_every_ms: float = 1_000.0
    """Frequency with which to check run progress in milliseconds."""
    as_agent: bool = False
    """Use as a LangChain agent, compatible with the AgentExecutor."""

    @model_validator(mode="after")
    def validate_async_client(self) -> Self:
        """Validate that the async client is set, otherwise initialize it."""
        if self.async_client is None:
            import openai

            api_key = self.client.api_key
            self.async_client = openai.AsyncOpenAI(api_key=api_key)
        return self

    @classmethod
    def create_assistant(
        cls,
        name: str,
        instructions: str,
        tools: Sequence[Union[BaseTool, dict]],
        model: str,
        *,
        model_kwargs: dict[str, float] = {},
        client: Optional[Union[openai.OpenAI, openai.AzureOpenAI]] = None,
        tool_resources: Optional[Union[AssistantToolResources, dict, NotGiven]] = None,
        extra_body: Optional[object] = None,
        **kwargs: Any,
    ) -> OpenAIAssistantRunnable:
        """Create an OpenAI Assistant and instantiate the Runnable.

        Args:
            name (str): Assistant name.
            instructions (str): Assistant instructions.
            tools (Sequence[Union[BaseTool, dict]]): Assistant tools. Can be passed
                in OpenAI format or as BaseTools.
            tool_resources (Optional[Union[AssistantToolResources, dict, NotGiven]]):
                Assistant tool resources. Can be passed in OpenAI format.
            model (str): Assistant model to use.
            client (Optional[Union[openai.OpenAI, openai.AzureOpenAI]]): OpenAI or
                AzureOpenAI client. Will create default OpenAI client (Assistant v2)
                if not specified.
            model_kwargs: Additional model arguments. Only available for temperature
                and top_p parameters.
            extra_body: Additional body parameters to be passed to the assistant.

        Returns:
            OpenAIAssistantRunnable: The configured assistant runnable.
        """
        client = client or _get_openai_client()
        if tool_resources is None:
            from openai._types import NOT_GIVEN

            tool_resources = NOT_GIVEN
        assistant = client.beta.assistants.create(
            name=name,
            instructions=instructions,
            tools=[_get_assistants_tool(tool) for tool in tools],  # type: ignore
            tool_resources=tool_resources,  # type: ignore[arg-type]
            model=model,
            extra_body=extra_body,
            **model_kwargs,
        )
        return cls(assistant_id=assistant.id, client=client, **kwargs)

    def invoke(
        self, input: dict, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> OutputType:
        """Invoke the assistant.

        Args:
            input (dict): Runnable input dict that can have:
                content: User message when starting a new run.
                thread_id: Existing thread to use.
                run_id: Existing run to use. Should only be supplied when providing
                    the tool output for a required action after an initial invocation.
                file_ids: (deprecated) File ids to include in new run. Use
                    'attachments' instead
                attachments: Assistant files to include in new run. (v2 API).
                message_metadata: Metadata to associate with new message.
                thread_metadata: Metadata to associate with new thread. Only relevant
                    when new thread being created.
                instructions: Additional run instructions.
                model: Override Assistant model for this run.
                tools: Override Assistant tools for this run.
                tool_resources: Override Assistant tool resources for this run (v2 API).
                run_metadata: Metadata to associate with new run.
            config (Optional[RunnableConfig]): Configuration for the run.

        Returns:
            OutputType: If self.as_agent, will return
                Union[List[OpenAIAssistantAction], OpenAIAssistantFinish]. Otherwise,
                will return OpenAI types
                Union[List[ThreadMessage], List[RequiredActionFunctionToolCall]].

        Raises:
            BaseException: If an error occurs during the invocation.
        """
        config = ensure_config(config)
        callback_manager = CallbackManager.configure(
            inheritable_callbacks=config.get("callbacks"),
            inheritable_tags=config.get("tags"),
            inheritable_metadata=config.get("metadata"),
        )
        run_manager = callback_manager.on_chain_start(
            dumpd(self), input, name=config.get("run_name") or self.get_name()
        )

        files = _convert_file_ids_into_attachments(kwargs.get("file_ids", []))
        attachments = kwargs.get("attachments", []) + files

        try:
            # Being run within AgentExecutor and there are tool outputs to submit.
            if self.as_agent and input.get("intermediate_steps"):
                tool_outputs = self._parse_intermediate_steps(
                    input["intermediate_steps"]
                )
                run = self.client.beta.threads.runs.submit_tool_outputs(**tool_outputs)
            # Starting a new thread and a new run.
            elif "thread_id" not in input:
                thread = {
                    "messages": [
                        {
                            "role": "user",
                            "content": input["content"],
                            "attachments": attachments,
                            "metadata": input.get("message_metadata"),
                        }
                    ],
                    "metadata": input.get("thread_metadata"),
                }
                run = self._create_thread_and_run(input, thread)
            # Starting a new run in an existing thread.
            elif "run_id" not in input:
                _ = self.client.beta.threads.messages.create(
                    input["thread_id"],
                    content=input["content"],
                    role="user",
                    attachments=attachments,
                    metadata=input.get("message_metadata"),
                )
                run = self._create_run(input)
            # Submitting tool outputs to an existing run, outside the AgentExecutor
            # framework.
            else:
                run = self.client.beta.threads.runs.submit_tool_outputs(**input)
            run = self._wait_for_run(run.id, run.thread_id)
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise e
        try:
            response = self._get_response(run)
        except BaseException as e:
            run_manager.on_chain_error(e, metadata=run.dict())
            raise e
        else:
            run_manager.on_chain_end(response)
            return response

    @classmethod
    async def acreate_assistant(
        cls,
        name: str,
        instructions: str,
        tools: Sequence[Union[BaseTool, dict]],
        model: str,
        *,
        async_client: Optional[
            Union[openai.AsyncOpenAI, openai.AsyncAzureOpenAI]
        ] = None,
        tool_resources: Optional[Union[AssistantToolResources, dict, NotGiven]] = None,
        **kwargs: Any,
    ) -> OpenAIAssistantRunnable:
        """Create an AsyncOpenAI Assistant and instantiate the Runnable.

        Args:
            name (str): Assistant name.
            instructions (str): Assistant instructions.
            tools (Sequence[Union[BaseTool, dict]]): Assistant tools. Can be passed
                in OpenAI format or as BaseTools.
            tool_resources (Optional[Union[AssistantToolResources, dict, NotGiven]]):
                Assistant tool resources. Can be passed in OpenAI format.
            model (str): Assistant model to use.
            async_client (Optional[Union[openai.OpenAI, openai.AzureOpenAI]]): OpenAI or
            AzureOpenAI async client. Will create default async_client if not specified.

        Returns:
            AsyncOpenAIAssistantRunnable: The configured assistant runnable.
        """
        async_client = async_client or _get_openai_async_client()
        if tool_resources is None:
            from openai._types import NOT_GIVEN

            tool_resources = NOT_GIVEN
        openai_tools = [_get_assistants_tool(tool) for tool in tools]

        assistant = await async_client.beta.assistants.create(
            name=name,
            instructions=instructions,
            tools=openai_tools,  # type: ignore
            tool_resources=tool_resources,  # type: ignore[arg-type]
            model=model,
        )
        return cls(assistant_id=assistant.id, async_client=async_client, **kwargs)

    async def ainvoke(
        self, input: dict, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> OutputType:
        """Async invoke assistant.

        Args:
            input (dict): Runnable input dict that can have:
                content: User message when starting a new run.
                thread_id: Existing thread to use.
                run_id: Existing run to use. Should only be supplied when providing
                    the tool output for a required action after an initial invocation.
                file_ids: (deprecated) File ids to include in new run. Use
                    'attachments' instead
                attachments: Assistant files to include in new run. (v2 API).
                message_metadata: Metadata to associate with new message.
                thread_metadata: Metadata to associate with new thread. Only relevant
                    when new thread being created.
                instructions: Additional run instructions.
                model: Override Assistant model for this run.
                tools: Override Assistant tools for this run.
                tool_resources: Override Assistant tool resources for this run (v2 API).
                run_metadata: Metadata to associate with new run.
            config (Optional[RunnableConfig]): Configuration for the run.

        Returns:
            OutputType: If self.as_agent, will return
                Union[List[OpenAIAssistantAction], OpenAIAssistantFinish]. Otherwise,
                will return OpenAI types
                Union[List[ThreadMessage], List[RequiredActionFunctionToolCall]].

        Raises:
            BaseException: If an error occurs during the invocation.
        """
        config = config or {}
        callback_manager = CallbackManager.configure(
            inheritable_callbacks=config.get("callbacks"),
            inheritable_tags=config.get("tags"),
            inheritable_metadata=config.get("metadata"),
        )
        run_manager = callback_manager.on_chain_start(
            dumpd(self), input, name=config.get("run_name") or self.get_name()
        )

        files = _convert_file_ids_into_attachments(kwargs.get("file_ids", []))
        attachments = kwargs.get("attachments", []) + files

        try:
            # Being run within AgentExecutor and there are tool outputs to submit.
            if self.as_agent and input.get("intermediate_steps"):
                tool_outputs = self._parse_intermediate_steps(
                    input["intermediate_steps"]
                )
                run = await self.async_client.beta.threads.runs.submit_tool_outputs(
                    **tool_outputs
                )
            # Starting a new thread and a new run.
            elif "thread_id" not in input:
                thread = {
                    "messages": [
                        {
                            "role": "user",
                            "content": input["content"],
                            "attachments": attachments,
                            "metadata": input.get("message_metadata"),
                        }
                    ],
                    "metadata": input.get("thread_metadata"),
                }
                run = await self._acreate_thread_and_run(input, thread)
            # Starting a new run in an existing thread.
            elif "run_id" not in input:
                _ = await self.async_client.beta.threads.messages.create(
                    input["thread_id"],
                    content=input["content"],
                    role="user",
                    attachments=attachments,
                    metadata=input.get("message_metadata"),
                )
                run = await self._acreate_run(input)
            # Submitting tool outputs to an existing run, outside the AgentExecutor
            # framework.
            else:
                run = await self.async_client.beta.threads.runs.submit_tool_outputs(
                    **input
                )
            run = await self._await_for_run(run.id, run.thread_id)
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise e
        try:
            response = self._get_response(run)
        except BaseException as e:
            run_manager.on_chain_error(e, metadata=run.dict())
            raise e
        else:
            run_manager.on_chain_end(response)
            return response

    def _create_run(self, input: dict) -> Any:
        """Create a new run within an existing thread.

        Args:
            input (dict): The input data for the new run.

        Returns:
            Any: The created run object.
        """
        allowed_assistant_params = (
            "instructions",
            "model",
            "tools",
            "tool_resources",
            "run_metadata",
            "truncation_strategy",
            "max_prompt_tokens",
        )
        params = {k: v for k, v in input.items() if k in allowed_assistant_params}
        return self.client.beta.threads.runs.create(
            input["thread_id"],
            assistant_id=self.assistant_id,
            **params,
        )

    def _create_thread_and_run(self, input: dict, thread: dict) -> Any:
        """Create a new thread and run.

        Args:
            input (dict): The input data for the run.
            thread (dict): The thread data to create.

        Returns:
            Any: The created thread and run.
        """
        params = {
            k: v
            for k, v in input.items()
            if k in ("instructions", "model", "tools", "run_metadata")
        }
        if tool_resources := input.get("tool_resources"):
            thread["tool_resources"] = tool_resources
        run = self.client.beta.threads.create_and_run(
            assistant_id=self.assistant_id,
            thread=thread,
            **params,
        )
        return run

    async def _acreate_run(self, input: dict) -> Any:
        """Asynchronously create a new run within an existing thread.

        Args:
            input (dict): The input data for the new run.

        Returns:
            Any: The created run object.
        """
        params = {
            k: v
            for k, v in input.items()
            if k in ("instructions", "model", "tools", "tool_resources", "run_metadata")
        }
        return await self.async_client.beta.threads.runs.create(
            input["thread_id"],
            assistant_id=self.assistant_id,
            **params,
        )

    async def _acreate_thread_and_run(self, input: dict, thread: dict) -> Any:
        """Asynchronously create a new thread and run simultaneously.

        Args:
            input (dict): The input data for the run.
            thread (dict): The thread data to create.

        Returns:
            Any: The created thread and run.
        """
        params = {
            k: v
            for k, v in input.items()
            if k in ("instructions", "model", "tools", "run_metadata")
        }
        if tool_resources := input.get("tool_resources"):
            thread["tool_resources"] = tool_resources
        run = await self.async_client.beta.threads.create_and_run(
            assistant_id=self.assistant_id,
            thread=thread,
            **params,
        )
        return run
