import json
import re
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

from langchain import LLMChain
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import Callbacks
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.tools.base import BaseTool

from langchain_experimental.pydantic_v1 import BaseModel

DEMONSTRATIONS = [
    {
        "role": "user",
        "content": "please show me a video and an image of (based on the text) 'a boy is running' and dub it",  # noqa: E501
    },
    {
        "role": "assistant",
        "content": '[{{"task": "video_generator", "id": 0, "dep": [-1], "args": {{"prompt": "a boy is running" }}}}, {{"task": "text_reader", "id": 1, "dep": [-1], "args": {{"text": "a boy is running" }}}}, {{"task": "image_generator", "id": 2, "dep": [-1], "args": {{"prompt": "a boy is running" }}}}]',  # noqa: E501
    },
    {
        "role": "user",
        "content": "Give you some pictures e1.jpg, e2.png, e3.jpg, help me count the number of sheep?",  # noqa: E501
    },
    {
        "role": "assistant",
        "content": '[ {{"task": "image_qa", "id": 0, "dep": [-1], "args": {{"image": "e1.jpg", "question": "How many sheep in the picture"}}}}, {{"task": "image_qa", "id": 1, "dep": [-1], "args": {{"image": "e2.jpg", "question": "How many sheep in the picture"}}}}, {{"task": "image_qa", "id": 2, "dep": [-1], "args": {{"image": "e3.jpg", "question": "How many sheep in the picture"}}}}]',  # noqa: E501
    },
]


class TaskPlaningChain(LLMChain):
    """Chain to execute tasks."""

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        demos: List[Dict] = DEMONSTRATIONS,
        verbose: bool = True,
    ) -> LLMChain:
        """Get the response parser."""
        system_template = """#1 Task Planning Stage: The AI assistant can parse user input to several tasks: [{{"task": task, "id": task_id, "dep": dependency_task_id, "args": {{"input name": text may contain <resource-dep_id>}}}}]. The special tag "dep_id" refer to the one generated text/image/audio in the dependency task (Please consider whether the dependency task generates resources of this type.) and "dep_id" must be in "dep" list. The "dep" field denotes the ids of the previous prerequisite tasks which generate a new resource that the current task relies on. The task MUST be selected from the following tools (along with tool description, input name and output type): {tools}. There may be multiple tasks of the same type. Think step by step about all the tasks needed to resolve the user's request. Parse out as few tasks as possible while ensuring that the user request can be resolved. Pay attention to the dependencies and order among tasks. If the user input can't be parsed, you need to reply empty JSON []."""  # noqa: E501
        human_template = """Now I input: {input}."""
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            system_template
        )
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        demo_messages: List[
            Union[HumanMessagePromptTemplate, AIMessagePromptTemplate]
        ] = []
        for demo in demos:
            if demo["role"] == "user":
                demo_messages.append(
                    HumanMessagePromptTemplate.from_template(demo["content"])
                )
            else:
                demo_messages.append(
                    AIMessagePromptTemplate.from_template(demo["content"])
                )
            # demo_messages.append(message)

        prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, *demo_messages, human_message_prompt]
        )

        return cls(prompt=prompt, llm=llm, verbose=verbose)


class Step:
    def __init__(
        self, task: str, id: int, dep: List[int], args: Dict[str, str], tool: BaseTool
    ):
        self.task = task
        self.id = id
        self.dep = dep
        self.args = args
        self.tool = tool


class Plan:
    def __init__(self, steps: List[Step]):
        self.steps = steps

    def __str__(self) -> str:
        return str([str(step) for step in self.steps])

    def __repr__(self) -> str:
        return str(self)


class BasePlanner(BaseModel):
    @abstractmethod
    def plan(self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any) -> Plan:
        """Given input, decide what to do."""

    @abstractmethod
    async def aplan(
        self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any
    ) -> Plan:
        """Given input, decide what to do."""


class PlanningOutputParser(BaseModel):
    def parse(self, text: str, hf_tools: List[BaseTool]) -> Plan:
        steps = []
        for v in json.loads(re.findall(r"\[.*\]", text)[0]):
            choose_tool = None
            for tool in hf_tools:
                if tool.name == v["task"]:
                    choose_tool = tool
                    break
            if choose_tool:
                steps.append(Step(v["task"], v["id"], v["dep"], v["args"], tool))
        return Plan(steps=steps)


class TaskPlanner(BasePlanner):
    llm_chain: LLMChain
    output_parser: PlanningOutputParser
    stop: Optional[List] = None

    def plan(self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any) -> Plan:
        """Given input, decided what to do."""
        inputs["tools"] = [
            f"{tool.name}: {tool.description}" for tool in inputs["hf_tools"]
        ]
        llm_response = self.llm_chain.run(**inputs, stop=self.stop, callbacks=callbacks)
        return self.output_parser.parse(llm_response, inputs["hf_tools"])

    async def aplan(
        self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any
    ) -> Plan:
        """Given input, decided what to do."""
        inputs["hf_tools"] = [
            f"{tool.name}: {tool.description}" for tool in inputs["hf_tools"]
        ]
        llm_response = await self.llm_chain.arun(
            **inputs, stop=self.stop, callbacks=callbacks
        )
        return self.output_parser.parse(llm_response, inputs["hf_tools"])


def load_chat_planner(llm: BaseLanguageModel) -> TaskPlanner:
    llm_chain = TaskPlaningChain.from_llm(llm)
    return TaskPlanner(llm_chain=llm_chain, output_parser=PlanningOutputParser())
