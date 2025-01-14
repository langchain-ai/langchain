from __future__ import annotations

import asyncio
import json
from json import JSONDecodeError
from time import sleep
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import CallbackManager
from langchain_core.load import dumpd
from langchain_core.runnables import RunnableConfig, RunnableSerializable, ensure_config
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

if TYPE_CHECKING:
    import openai
    from openai.types.beta.threads import ThreadMessage
    from openai.types.beta.threads.required_action_function_tool_call import (
        RequiredActionFunctionToolCall,
    )


class OpenAIAssistantFinish(AgentFinish):
    """AgentFinish with run and thread metadata.

    Parameters:
        run_id: Run id.
        thread_id: Thread id.
    """

    run_id: str
    thread_id: str

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Check if the class is serializable by LangChain.

        Returns:
            False
        """
        return False


class OpenAIAssistantAction(AgentAction):
    """AgentAction with info needed to submit custom tool output to existing run.

    Parameters:
        tool_call_id: Tool call id.
        run_id: Run id.
        thread_id: Thread id
    """

    tool_call_id: str
    run_id: str
    thread_id: str

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Check if the class is serializable by LangChain.

        Returns:
            False
        """
        return False


def _get_openai_client() -> openai.OpenAI:
    try:
        import openai

        return openai.OpenAI()
    except ImportError as e:
        raise ImportError(
            "Unable to import openai, please install with `pip install openai`."
        ) from e
    except AttributeError as e:
        raise AttributeError(
            "Please make sure you are using a v1.1-compatible version of openai. You "
            'can install with `pip install "openai>=1.1"`.'
        ) from e


def _get_openai_async_client() -> openai.AsyncOpenAI:
    try:
        import openai

        return openai.AsyncOpenAI()
    except ImportError as e:
        raise ImportError(
            "Unable to import openai, please install with `pip install openai`."
        ) from e
    except AttributeError as e:
        raise AttributeError(
            "Please make sure you are using a v1.1-compatible version of openai. You "
            'can install with `pip install "openai>=1.1"`.'
        ) from e


def _is_assistants_builtin_tool(
    tool: Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool],
) -> bool:
    """Determine if tool corresponds to OpenAI Assistants built-in."""
    assistants_builtin_tools = ("code_interpreter", "retrieval")
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
    """
    if _is_assistants_builtin_tool(tool):
        return tool  # type: ignore
    else:
        return convert_to_openai_tool(tool)


OutputType = Union[
    List[OpenAIAssistantAction],
    OpenAIAssistantFinish,
    List["ThreadMessage"],
    List["RequiredActionFunctionToolCall"],
]


class OpenAIAssistantRunnable(RunnableSerializable[Dict, OutputType]):
    """Run an OpenAI Assistant.

    Example using OpenAI tools:
        .. code-block:: python

            from langchain_experimental.openai_assistant import OpenAIAssistantRunnable

            interpreter_assistant = OpenAIAssistantRunnable.create_assistant(
                name="langchain assistant",
                instructions="You are a personal math tutor. Write and run code to answer math questions.",
                tools=[{"type": "code_interpreter"}],
                model="gpt-4-1106-preview"
            )
            output = interpreter_assistant.invoke({"content": "What's 10 - 4 raised to the 2.7"})

    Example using custom tools and AgentExecutor:
        .. code-block:: python

            from langchain_experimental.openai_assistant import OpenAIAssistantRunnable
            from langchain.agents import AgentExecutor
            from langchain.tools import E2BDataAnalysisTool


            tools = [E2BDataAnalysisTool(api_key="...")]
            agent = OpenAIAssistantRunnable.create_assistant(
                name="langchain assistant e2b tool",
                instructions="You are a personal math tutor. Write and run code to answer math questions.",
                tools=tools,
                model="gpt-4-1106-preview",
                as_agent=True
            )

            agent_executor = AgentExecutor(agent=agent, tools=tools)
            agent_executor.invoke({"content": "What's 10 - 4 raised to the 2.7"})


    Example using custom tools and custom execution:
        .. code-block:: python

            from langchain_experimental.openai_assistant import OpenAIAssistantRunnable
            from langchain.agents import AgentExecutor
            from langchain_core.agents import AgentFinish
            from langchain.tools import E2BDataAnalysisTool


            tools = [E2BDataAnalysisTool(api_key="...")]
            agent = OpenAIAssistantRunnable.create_assistant(
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
    """Frequency with which to check run progress in ms."""
    as_agent: bool = False
    """Use as a LangChain agent, compatible with the AgentExecutor."""

    @model_validator(mode="after")
    def validate_async_client(self) -> Self:
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
        client: Optional[Union[openai.OpenAI, openai.AzureOpenAI]] = None,
        **kwargs: Any,
    ) -> OpenAIAssistantRunnable:
        """Create an OpenAI Assistant and instantiate the Runnable.

        Args:
            name: Assistant name.
            instructions: Assistant instructions.
            tools: Assistant tools. Can be passed in OpenAI format or as BaseTools.
            model: Assistant model to use.
            client: OpenAI or AzureOpenAI client.
                Will create a default OpenAI client if not specified.
            kwargs: Additional arguments.

        Returns:
            OpenAIAssistantRunnable configured to run using the created assistant.
        """
        client = client or _get_openai_client()
        assistant = client.beta.assistants.create(
            name=name,
            instructions=instructions,
            tools=[_get_assistants_tool(tool) for tool in tools],  # type: ignore
            model=model,
        )
        return cls(assistant_id=assistant.id, client=client, **kwargs)

    def invoke(
        self, input: dict, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> OutputType:
        """Invoke assistant.

        Args:
            input: Runnable input dict that can have:
                content: User message when starting a new run.
                thread_id: Existing thread to use.
                run_id: Existing run to use. Should only be supplied when providing
                    the tool output for a required action after an initial invocation.
                message_metadata: Metadata to associate with new message.
                thread_metadata: Metadata to associate with new thread. Only relevant
                    when new thread being created.
                instructions: Additional run instructions.
                model: Override Assistant model for this run.
                tools: Override Assistant tools for this run.
                parallel_tool_calls: Allow Assistant to set parallel_tool_calls
                    for this run.
                top_p: Override Assistant top_p for this run.
                temperature: Override Assistant temperature for this run.
                max_completion_tokens: Allow setting max_completion_tokens for this run.
                max_prompt_tokens: Allow setting max_prompt_tokens for this run.
                run_metadata: Metadata to associate with new run.
            config: Runnable config. Defaults to None.

        Return:
            If self.as_agent, will return
                Union[List[OpenAIAssistantAction], OpenAIAssistantFinish].
                Otherwise, will return OpenAI types
                Union[List[ThreadMessage], List[RequiredActionFunctionToolCall]].
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
        **kwargs: Any,
    ) -> OpenAIAssistantRunnable:
        """Async create an AsyncOpenAI Assistant and instantiate the Runnable.

        Args:
            name: Assistant name.
            instructions: Assistant instructions.
            tools: Assistant tools. Can be passed in OpenAI format or as BaseTools.
            model: Assistant model to use.
            async_client: AsyncOpenAI client.
                Will create default async_client if not specified.

        Returns:
            AsyncOpenAIAssistantRunnable configured to run using the created assistant.
        """
        async_client = async_client or _get_openai_async_client()
        openai_tools = [_get_assistants_tool(tool) for tool in tools]
        assistant = await async_client.beta.assistants.create(
            name=name,
            instructions=instructions,
            tools=openai_tools,  # type: ignore
            model=model,
        )
        return cls(assistant_id=assistant.id, async_client=async_client, **kwargs)

    async def ainvoke(
        self, input: dict, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> OutputType:
        """Async invoke assistant.

        Args:
            input: Runnable input dict that can have:
                content: User message when starting a new run.
                thread_id: Existing thread to use.
                run_id: Existing run to use. Should only be supplied when providing
                    the tool output for a required action after an initial invocation.
                message_metadata: Metadata to associate with a new message.
                thread_metadata: Metadata to associate with new thread. Only relevant
                    when a new thread is created.
                instructions: Overrides the instructions of the assistant.
                additional_instructions: Appends additional instructions.
                model: Override Assistant model for this run.
                tools: Override Assistant tools for this run.
                parallel_tool_calls: Allow Assistant to set parallel_tool_calls
                    for this run.
                top_p: Override Assistant top_p for this run.
                temperature: Override Assistant temperature for this run.
                max_completion_tokens: Allow setting max_completion_tokens for this run.
                max_prompt_tokens: Allow setting max_prompt_tokens for this run.
                run_metadata: Metadata to associate with new run.
            config: Runnable config. Defaults to None.
            kwargs: Additional arguments.

        Return:
            If self.as_agent, will return
                Union[List[OpenAIAssistantAction], OpenAIAssistantFinish].
                Otherwise, will return OpenAI types
                Union[List[ThreadMessage], List[RequiredActionFunctionToolCall]].
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
        try:
            # Being run within AgentExecutor and there are tool outputs to submit.
            if self.as_agent and input.get("intermediate_steps"):
                tool_outputs = await self._aparse_intermediate_steps(
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

    def _parse_intermediate_steps(
        self, intermediate_steps: List[Tuple[OpenAIAssistantAction, str]]
    ) -> dict:
        last_action, last_output = intermediate_steps[-1]
        run = self._wait_for_run(last_action.run_id, last_action.thread_id)
        required_tool_call_ids = set()
        if run.required_action:
            required_tool_call_ids = {
                tc.id for tc in run.required_action.submit_tool_outputs.tool_calls
            }
        tool_outputs = [
            {"output": str(output), "tool_call_id": action.tool_call_id}
            for action, output in intermediate_steps
            if action.tool_call_id in required_tool_call_ids
        ]
        submit_tool_outputs = {
            "tool_outputs": tool_outputs,
            "run_id": last_action.run_id,
            "thread_id": last_action.thread_id,
        }
        return submit_tool_outputs

    def _create_run(self, input: dict) -> Any:
        params = {
            k: v
            for k, v in input.items()
            if k
            in (
                "instructions",
                "model",
                "tools",
                "additional_instructions",
                "parallel_tool_calls",
                "top_p",
                "temperature",
                "max_completion_tokens",
                "max_prompt_tokens",
                "run_metadata",
            )
        }
        return self.client.beta.threads.runs.create(
            input["thread_id"],
            assistant_id=self.assistant_id,
            **params,
        )

    def _create_thread_and_run(self, input: dict, thread: dict) -> Any:
        params = {
            k: v
            for k, v in input.items()
            if k
            in (
                "instructions",
                "model",
                "tools",
                "parallel_tool_calls",
                "top_p",
                "temperature",
                "max_completion_tokens",
                "max_prompt_tokens",
                "run_metadata",
            )
        }
        run = self.client.beta.threads.create_and_run(
            assistant_id=self.assistant_id,
            thread=thread,
            **params,
        )
        return run

    def _get_response(self, run: Any) -> Any:
        # TODO: Pagination

        if run.status == "completed":
            import openai

            major_version = int(openai.version.VERSION.split(".")[0])
            minor_version = int(openai.version.VERSION.split(".")[1])
            version_gte_1_14 = (major_version > 1) or (
                major_version == 1 and minor_version >= 14
            )

            messages = self.client.beta.threads.messages.list(
                run.thread_id, order="asc"
            )
            new_messages = [msg for msg in messages if msg.run_id == run.id]
            if not self.as_agent:
                return new_messages
            answer: Any = [
                msg_content for msg in new_messages for msg_content in msg.content
            ]
            if all(
                (
                    isinstance(content, openai.types.beta.threads.TextContentBlock)
                    if version_gte_1_14
                    else isinstance(
                        content, openai.types.beta.threads.MessageContentText
                    )
                )
                for content in answer
            ):
                answer = "\n".join(content.text.value for content in answer)
            return OpenAIAssistantFinish(
                return_values={
                    "output": answer,
                    "thread_id": run.thread_id,
                    "run_id": run.id,
                },
                log="",
                run_id=run.id,
                thread_id=run.thread_id,
            )
        elif run.status == "requires_action":
            if not self.as_agent:
                return run.required_action.submit_tool_outputs.tool_calls
            actions = []
            for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                function = tool_call.function
                try:
                    args = json.loads(function.arguments, strict=False)
                except JSONDecodeError as e:
                    raise ValueError(
                        f"Received invalid JSON function arguments: "
                        f"{function.arguments} for function {function.name}"
                    ) from e
                if len(args) == 1 and "__arg1" in args:
                    args = args["__arg1"]
                actions.append(
                    OpenAIAssistantAction(
                        tool=function.name,
                        tool_input=args,
                        tool_call_id=tool_call.id,
                        log="",
                        run_id=run.id,
                        thread_id=run.thread_id,
                    )
                )
            return actions
        else:
            run_info = json.dumps(run.dict(), indent=2)
            raise ValueError(
                f"Unexpected run status: {run.status}. Full run info:\n\n{run_info})"
            )

    def _wait_for_run(self, run_id: str, thread_id: str) -> Any:
        in_progress = True
        while in_progress:
            run = self.client.beta.threads.runs.retrieve(run_id, thread_id=thread_id)
            in_progress = run.status in ("in_progress", "queued")
            if in_progress:
                sleep(self.check_every_ms / 1000)
        return run

    async def _aparse_intermediate_steps(
        self, intermediate_steps: List[Tuple[OpenAIAssistantAction, str]]
    ) -> dict:
        last_action, last_output = intermediate_steps[-1]
        run = await self._wait_for_run(last_action.run_id, last_action.thread_id)
        required_tool_call_ids = set()
        if run.required_action:
            required_tool_call_ids = {
                tc.id for tc in run.required_action.submit_tool_outputs.tool_calls
            }
        tool_outputs = [
            {"output": str(output), "tool_call_id": action.tool_call_id}
            for action, output in intermediate_steps
            if action.tool_call_id in required_tool_call_ids
        ]
        submit_tool_outputs = {
            "tool_outputs": tool_outputs,
            "run_id": last_action.run_id,
            "thread_id": last_action.thread_id,
        }
        return submit_tool_outputs

    async def _acreate_run(self, input: dict) -> Any:
        params = {
            k: v
            for k, v in input.items()
            if k
            in (
                "instructions",
                "model",
                "tools",
                "additional_instructions",
                "parallel_tool_calls",
                "top_p",
                "temperature",
                "max_completion_tokens",
                "max_prompt_tokens",
                "run_metadata",
            )
        }
        return await self.async_client.beta.threads.runs.create(
            input["thread_id"],
            assistant_id=self.assistant_id,
            **params,
        )

    async def _acreate_thread_and_run(self, input: dict, thread: dict) -> Any:
        params = {
            k: v
            for k, v in input.items()
            if k
            in (
                "instructions",
                "model",
                "tools",
                "parallel_tool_calls",
                "top_p",
                "temperature",
                "max_completion_tokens",
                "max_prompt_tokens",
                "run_metadata",
            )
        }
        run = await self.async_client.beta.threads.create_and_run(
            assistant_id=self.assistant_id,
            thread=thread,
            **params,
        )
        return run

    async def _aget_response(self, run: Any) -> Any:
        # TODO: Pagination

        if run.status == "completed":
            import openai

            major_version = int(openai.version.VERSION.split(".")[0])
            minor_version = int(openai.version.VERSION.split(".")[1])
            version_gte_1_14 = (major_version > 1) or (
                major_version == 1 and minor_version >= 14
            )

            messages = await self.async_client.beta.threads.messages.list(
                run.thread_id, order="asc"
            )
            new_messages = [msg for msg in messages if msg.run_id == run.id]
            if not self.as_agent:
                return new_messages
            answer: Any = [
                msg_content for msg in new_messages for msg_content in msg.content
            ]
            if all(
                (
                    isinstance(content, openai.types.beta.threads.TextContentBlock)
                    if version_gte_1_14
                    else isinstance(
                        content, openai.types.beta.threads.MessageContentText
                    )
                )
                for content in answer
            ):
                answer = "\n".join(content.text.value for content in answer)
            return OpenAIAssistantFinish(
                return_values={
                    "output": answer,
                    "thread_id": run.thread_id,
                    "run_id": run.id,
                },
                log="",
                run_id=run.id,
                thread_id=run.thread_id,
            )
        elif run.status == "requires_action":
            if not self.as_agent:
                return run.required_action.submit_tool_outputs.tool_calls
            actions = []
            for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                function = tool_call.function
                try:
                    args = json.loads(function.arguments, strict=False)
                except JSONDecodeError as e:
                    raise ValueError(
                        f"Received invalid JSON function arguments: "
                        f"{function.arguments} for function {function.name}"
                    ) from e
                if len(args) == 1 and "__arg1" in args:
                    args = args["__arg1"]
                actions.append(
                    OpenAIAssistantAction(
                        tool=function.name,
                        tool_input=args,
                        tool_call_id=tool_call.id,
                        log="",
                        run_id=run.id,
                        thread_id=run.thread_id,
                    )
                )
            return actions
        else:
            run_info = json.dumps(run.dict(), indent=2)
            raise ValueError(
                f"Unexpected run status: {run.status}. Full run info:\n\n{run_info})"
            )

    async def _await_for_run(self, run_id: str, thread_id: str) -> Any:
        in_progress = True
        while in_progress:
            run = await self.async_client.beta.threads.runs.retrieve(
                run_id, thread_id=thread_id
            )
            in_progress = run.status in ("in_progress", "queued")
            if in_progress:
                await asyncio.sleep(self.check_every_ms / 1000)
        return run
