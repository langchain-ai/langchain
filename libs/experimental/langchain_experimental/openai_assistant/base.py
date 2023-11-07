from __future__ import annotations

import json
from time import sleep
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

from langchain.pydantic_v1 import root_validator
from langchain.schema.agent import AgentAction, AgentFinish
from langchain.schema.runnable import RunnableConfig, RunnableSerializable

if TYPE_CHECKING:
    import openai


class OpenAIAssistantFinish(AgentFinish):
    run_id: str
    thread_id: str


class OpenAIAssistantAction(AgentAction):
    tool_call_id: str
    run_id: str
    thread_id: str


class OpenAIAssistantRunnable(RunnableSerializable[Union[List[dict], str], list]):
    client: Optional[openai.OpenAI] = None
    assistant_id: str
    check_every_ms: float = 1_000.0
    as_agent: bool = False

    @classmethod
    def create(
        cls,
        name: str,
        instructions: str,
        tools: Sequence,
        model: str,
        *,
        client: Optional[openai.OpenAI] = None,
        **kwargs: Any,
    ) -> OpenAIAssistantRunnable:
        assistant = client.beta.assistants.create(
            name=name,
            instructions=instructions,
            tools=tools,
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

    def invoke(self, input: dict, config: Optional[RunnableConfig] = None) -> List:
        input = self._parse_input(input)
        if "thread_id" not in input:
            run = self._create_thread_and_run(input)
            _ = self.client.beta.threads.messages.create(
                run.thread_id, content=input["content"], role="user"
            )
        elif "run_id" not in input:
            _ = self.client.beta.threads.messages.create(
                input["thread_id"], content=input["content"], role="user"
            )
            run = self._create_run(input)
        else:
            run = self.client.beta.threads.runs.submit_tool_outputs(
                thread_id=input["thread_id"], **input
            )
        return self._get_response(run.id)

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
            if k in ("instructions", "model", "tools", "metadata")
        }
        return self.client.beta.threads.runs.create(
            input["thread_id"],
            assistant_id=self.assistant_id,
            **params,
        )

    def _create_thread_and_run(self, input: dict) -> Any:
        params = {
            k: v
            for k, v in input.items()
            if k in ("instructions", "thread", "model", "tools", "metadata")
        }
        run = self.client.beta.threads.create_and_run(
            assistant_id=self.assistant_id,
            **params,
        )
        return run

    def _get_response(
        self, run_id: str, thread_id: str
    ) -> Union[List[OpenAIAssistantAction], OpenAIAssistantFinish]:
        # TODO: Pagination
        run = self._wait_for_run(run_id, thread_id)
        if run.status == "completed":
            messages = self.client.beta.threads.messages.list(
                self.thread_id, order="asc"
            )
            new_messages = [msg for msg in messages if msg.run_id == run_id]
            if not self.as_agent:
                return new_messages
            answer = "".join(
                msg_content.text.value
                for msg in new_messages
                for msg_content in msg.content
            )
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

    def _wait_for_run(self, run_id: str, thread_id) -> Any:
        in_progress = True
        while in_progress:
            run = self.client.beta.threads.runs.retrieve(run_id, thread_id=thread_id)
            in_progress = run.status in ("in_progress", "queued")
            if in_progress:
                sleep(self.check_every_ms / 1000)
        return run
