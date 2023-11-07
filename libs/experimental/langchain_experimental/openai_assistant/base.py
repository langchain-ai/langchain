from __future__ import annotations

import json
from time import sleep
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from langchain.pydantic_v1 import root_validator
from langchain.schema.agent import AgentAction, AgentFinish
from langchain.schema.runnable import RunnableConfig, RunnableSerializable

if TYPE_CHECKING:
    import openai


class OpenAIAssistantFinish(AgentFinish):
    run_id: str


class OpenAIAssistantAction(AgentAction):
    tool_call_id: str
    run_id: str


class OpenAIAssistantRunnable(RunnableSerializable[Union[List[dict], str], list]):
    client: Optional[openai.OpenAI] = None
    name: str
    instructions: str
    tools: list
    model: str
    thread_id: Optional[str] = None
    assistant_id: Optional[str] = None
    run_id: Optional[str] = None
    poll_rate: int = 1
    as_agent: bool = False

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        if not values["client"]:
            try:
                import openai
            except ImportError as e:
                raise ImportError() from e

            values["client"] = openai.OpenAI()
        return values

    def create(self) -> None:
        if not self.assistant_id:
            assistant = self.client.beta.assistants.create(
                name=self.name,
                instructions=self.instructions,
                tools=self.tools,
                model=self.model,
            )
            self.assistant_id = assistant.id
        if not self.thread_id:
            thread = self.client.beta.threads.create()
            self.thread_id = thread.id

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> List:
        if "run_id" not in input:
            msg = self.client.beta.threads.messages.create(
                self.thread_id, content=input["content"], role="user"
            )
            run = self.client.beta.threads.runs.create(
                self.thread_id,
                assistant_id=self.assistant_id,
                instructions=input.get("run_instructions"),
            )

        else:
            run = self.client.beta.threads.runs.submit_tool_outputs(
                thread_id=self.thread_id, **input
            )
        return self._get_response(run.id)

    def _get_response(self, run_id: str) -> List:
        # TODO: Pagination
        in_progress = True
        while in_progress:
            run = self._retrieve_run(run_id)
            in_progress = run.status in ("in_progress", "queued")
            if in_progress:
                sleep(self.poll_rate)
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
                return_values={"output": answer}, log="", run_id=run_id
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
                    )
                )
            return actions
        else:
            raise ValueError(run.dict())

    def _retrieve_run(self, run_id: str) -> Any:
        return self.client.beta.threads.runs.retrieve(run_id, thread_id=self.thread_id)
