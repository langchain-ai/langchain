from __future__ import annotations

import json
from time import sleep
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

from langchain.pydantic_v1 import Field, root_validator
from langchain.schema.agent import AgentAction, AgentFinish
from langchain.schema.runnable import RunnableConfig, RunnableSerializable
from langchain.tools import format_tool_to_openai_function
from langchain.tools.base import BaseTool

if TYPE_CHECKING:
    import openai


class OpenAIAssistantFinish(AgentFinish):
    run_id: str
    thread_id: str


class OpenAIAssistantAction(AgentAction):
    tool_call_id: str
    run_id: str
    thread_id: str


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


class OpenAIAssistantRunnable(RunnableSerializable[Dict, Any]):
    """Run an OpenAI Assistant."""

    client: openai.OpenAI = Field(default_factory=_get_openai_client)
    """OpenAI client."""
    assistant_id: str
    """OpenAI assistant id."""
    check_every_ms: float = 1_000.0
    """Frequency with which to check run progress in ms."""
    as_agent: bool = False
    """Use as a LangChain agent, compatible with the AgentExecutor."""

    @classmethod
    def create_assistant(
        cls,
        name: str,
        instructions: str,
        tools: Sequence[Union[BaseTool, dict]],
        model: str,
        *,
        client: Optional[openai.OpenAI] = None,
        **kwargs: Any,
    ) -> OpenAIAssistantRunnable:
        """Create an OpenAI Assistant and instantiate the Runnable.

        Args:
            name: Assistant name.
            instructions: Assistant instructions.
            tools: Assistant tools. Can be passed in in OpenAI format or as BaseTools.
            model: Assistant model to use.
            client: OpenAI client. Will create default client if not specified.

        Returns:
            OpenAIAssistantRunnable configured to run using the created assistant.
        """
        client = client or _get_openai_client()
        openai_tools: List = []
        for tool in tools:
            if isinstance(tool, BaseTool):
                tool = {"type": "function", "function": format_tool_to_openai_function(tool)}
            openai_tools.append(tool)
        assistant = client.beta.assistants.create(
            name=name,
            instructions=instructions,
            tools=openai_tools,
            model=model,
        )
        return cls(assistant_id=assistant.id, **kwargs)

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        if not values["client"]:
            try:
                import openai
            except ImportError as e:
                raise ImportError() from e

            values["client"] = openai.OpenAI()
        return values

    def invoke(self, input: dict, config: Optional[RunnableConfig] = None) -> Any:
        """"""
        input = self._parse_input(input)
        if "thread_id" not in input:
            thread = {
                "messages": [
                    {
                        "role": "user",
                        "content": input["content"],
                        "file_ids": input.get("file_ids", []),
                        "metadata": input.get("message_metadata"),
                    }
                ],
                "metadata": input.get("thread_metadata"),
            }
            run = self._create_thread_and_run(input, thread)
        elif "run_id" not in input:
            _ = self.client.beta.threads.messages.create(
                input["thread_id"],
                content=input["content"],
                role="user",
                file_ids=input.get("file_ids", []),
                metadata=input.get("message_metadata"),
            )
            run = self._create_run(input)
        else:
            run = self.client.beta.threads.runs.submit_tool_outputs(**input)
        return self._get_response(run.id, run.thread_id)

    def _parse_input(self, input: dict) -> dict:
        if self.as_agent and input.get("intermediate_steps"):
            last_action, last_output = input["intermediate_steps"][-1]
            input = {
                "tool_outputs": [
                    {"output": last_output, "tool_call_id": last_action.tool_call_id}
                ],
                "run_id": last_action.run_id,
                "thread_id": last_action.thread_id,
            }
        return input

    def _create_run(self, input: dict) -> Any:
        params = {
            k: v
            for k, v in input.items()
            if k in ("instructions", "model", "tools", "run_metadata")
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
            if k in ("instructions", "model", "tools", "run_metadata")
        }
        run = self.client.beta.threads.create_and_run(
            assistant_id=self.assistant_id,
            thread=thread,
            **params,
        )
        return run

    def _get_response(self, run_id: str, thread_id: str) -> Any:
        # TODO: Pagination
        import openai

        run = self._wait_for_run(run_id, thread_id)
        if run.status == "completed":
            messages = self.client.beta.threads.messages.list(thread_id, order="asc")
            new_messages = [msg for msg in messages if msg.run_id == run_id]
            if not self.as_agent:
                return new_messages
            answer: Any = [
                msg_content for msg in new_messages for msg_content in msg.content
            ]
            if all(
                isinstance(content, openai.types.beta.threads.MessageContentText)
                for content in answer
            ):
                answer = "".join(content.text.value for content in answer)
            return OpenAIAssistantFinish(
                return_values={"output": answer},
                log="",
                run_id=run_id,
                thread_id=thread_id,
            )
        elif run.status == "requires_action":
            if not self.as_agent:
                return run.required_action.submit_tool_outputs
            actions = []
            for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                function = tool_call.function
                args = json.loads(function.arguments)
                actions.append(
                    OpenAIAssistantAction(
                        tool=function.name,
                        tool_input=args,
                        tool_call_id=tool_call.id,
                        log="",
                        run_id=run_id,
                        thread_id=thread_id,
                    )
                )
            return actions
        else:
            run_info = json.dumps(run.dict(), indent=2)
            raise ValueError(
                f"Unknown run status {run.status}. Full run info:\n\n{run_info})"
            )

    def _wait_for_run(self, run_id: str, thread_id: str) -> Any:
        in_progress = True
        while in_progress:
            run = self.client.beta.threads.runs.retrieve(run_id, thread_id=thread_id)
            in_progress = run.status in ("in_progress", "queued")
            if in_progress:
                sleep(self.check_every_ms / 1000)
        return run
