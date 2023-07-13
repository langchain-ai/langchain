import json
import re
from typing import Any, Dict, List, Optional
from langchain import LLMChain
from langchain.base_language import BaseLanguageModel
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.experimental.plan_and_execute.schema import PlanOutputParser
from langchain.experimental.plan_and_execute.planners.base import BasePlanner
from langchain.callbacks.manager import Callbacks

from transformers import load_tool

DEMONSTRATIONS = [
    {
        "role": "user",
        "content": "please show me a video and an image of (based on the text) 'a boy is running' and dub it"
    },
    {
        "role": "assistant",
        "content": "[{{\"task\": \"video_generator\", \"id\": 0, \"dep\": [-1], \"args\": {{\"prompt\": \"a boy is running\" }}}}, {{\"task\": \"text_reader\", \"id\": 1, \"dep\": [-1], \"args\": {{\"text\": \"a boy is running\" }}}}, {{\"task\": \"image_generator\", \"id\": 2, \"dep\": [-1], \"args\": {{\"prompt\": \"a boy is running\" }}}}]"
    }
]

class TaskPlaningChain(LLMChain):
    """Chain to execute tasks."""
    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, demos: List[Dict] = DEMONSTRATIONS, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        system_template = """#1 Task Planning Stage: The AI assistant can parse user input to several tasks: [{{"task": task, "id": task_id, "dep": dependency_task_id, "args": {{"input name": input value or dep_id}}}}]. The special tag "dep_id" refer to the one generated text/image/audio in the dependency task (Please consider whether the dependency task generates resources of this type.) and "dep_id" must be in "dep" list. The "dep" field denotes the ids of the previous prerequisite tasks which generate a new resource that the current task relies on. The task MUST be selected from the following tools (along with tool description, input name and output type): {tools}. There may be multiple tasks of the same type. Think step by step about all the tasks needed to resolve the user's request. Parse out as few tasks as possible while ensuring that the user request can be resolved. Pay attention to the dependencies and order among tasks. If the user input can't be parsed, you need to reply empty JSON []."""
        # human_template = """The chat log [ {context} ] may contain the resources I mentioned. Now I input: {input}."""
        human_template = """Now I input: {input}."""
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        demo_messages = []
        for demo in demos:
            if demo["role"] == "user":
                message = HumanMessagePromptTemplate.from_template(demo["content"])
            else:
                message = AIMessagePromptTemplate.from_template(demo["content"])
            demo_messages.append(message)

        prompt = ChatPromptTemplate.from_messages([system_message_prompt, *demo_messages, human_message_prompt])

        return cls(prompt=prompt, llm=llm, verbose=verbose)

class Step:
    def __init__(self, task: str, id: int, dep: List[int], args: Dict[str, str], tool):
        self.task = task
        self.id = id
        self.dep = dep
        self.args = args
        self.tool = tool

class Plan:
    def __init__(self, steps: List[Step]):
        self.steps = steps

    def __str__(self):
        return str([str(step) for step in self.steps])

    def __repr__(self):
        return str(self)

class PlanningOutputParser(PlanOutputParser):

    def parse(self, text: str, hf_tools) -> Plan:
        print(text)
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
    output_parser: PlanOutputParser
    stop: Optional[List] = None

    def plan(self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any) -> Plan:
        """Given input, decided what to do."""
        inputs["tools"] = [f"{tool.name}: {tool.description}" for tool in inputs["hf_tools"]]
        llm_response = self.llm_chain.run(**inputs, stop=self.stop, callbacks=callbacks)
        return self.output_parser.parse(llm_response, inputs["hf_tools"])

    async def aplan(
        self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any
    ) -> Plan:
        """Given input, decided what to do."""
        inputs["hf_tools"] = [f"{tool.name}: {tool.description}" for tool in inputs["hf_tools"]]
        llm_response = await self.llm_chain.arun(
            **inputs, stop=self.stop, callbacks=callbacks
        )
        return self.output_parser.parse(llm_response)


def load_chat_planner(llm: BaseLanguageModel) -> TaskPlanner:
    llm_chain = TaskPlaningChain.from_llm(llm)
    return TaskPlanner(
        llm_chain=llm_chain,
        output_parser=PlanningOutputParser()
    )