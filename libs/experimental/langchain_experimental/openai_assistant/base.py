from __future__ import annotations

from time import sleep
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from langchain.pydantic_v1 import root_validator
from langchain.schema.runnable import RunnableConfig, RunnableSerializable

if TYPE_CHECKING:
    import openai


class OpenAIAssistantRunnable(RunnableSerializable[Union[List[dict], str], list]):
    client: Optional[openai.OpenAI] = None
    name: str
    instructions: str
    tools: list
    model: str
    thread_id: Optional[str] = None
    assistant_id: Optional[str] = None
    run_id: Optional[str] = None
    run_instructions: Optional[str] = None

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        if not values["client"]:
            try:
                import openai
            except ImportError as e:
                raise ImportError() from e

            values["client"] = openai.OpenAI()
        return values

    def create(self, run_instructions: Optional[str] = None) -> None:
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

    def invoke(
        self, input: Union[str, List[dict]], config: Optional[RunnableConfig] = None
    ) -> List:
        if isinstance(input, str):
            msg = self.client.beta.threads.messages.create(
                self.thread_id, content=input, role="user"
            )
            run = self.client.beta.threads.runs.create(
                self.thread_id,
                assistant_id=self.assistant_id,
                instructions=self.run_instructions,
            )
            self.run_id = run.id

        else:
            run = self.client.beta.threads.runs.submit_tool_outputs(
                thread_id=self.thread_id,
                run_id=self.run_id,
                tool_outputs=input,
            )
        return self._list_steps()

    def _list_steps(self) -> List:
        # TODO: Pagination
        in_progress = True
        while in_progress:
            steps = self.client.beta.threads.runs.steps.list(
                self.run_id, thread_id=self.thread_id, order="asc"
            )
            in_progress = not steps.data or steps.data[-1].status == "in_progress"
            if in_progress:
                sleep(1)

        return steps.data
