from typing import List

from langchain.base_language import BaseLanguageModel
from langchain_core.tools import BaseTool

from langchain_experimental.autonomous_agents.hugginggpt.repsonse_generator import (
    load_response_generator,
)
from langchain_experimental.autonomous_agents.hugginggpt.task_executor import (
    TaskExecutor,
)
from langchain_experimental.autonomous_agents.hugginggpt.task_planner import (
    load_chat_planner,
)


class HuggingGPT:
    """Agent for interacting with HuggingGPT."""

    def __init__(self, llm: BaseLanguageModel, tools: List[BaseTool]):
        self.llm = llm
        self.tools = tools
        self.chat_planner = load_chat_planner(llm)
        self.response_generator = load_response_generator(llm)
        self.task_executor: TaskExecutor

    def run(self, input: str) -> str:
        plan = self.chat_planner.plan(inputs={"input": input, "hf_tools": self.tools})
        self.task_executor = TaskExecutor(plan)
        self.task_executor.run()
        response = self.response_generator.generate(
            {"task_execution": self.task_executor}
        )
        return response
