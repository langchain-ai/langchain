import json
import re
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

from langchain.base_language import BaseLanguageModel
from langchain.chains import LLMChain
from langchain_core.callbacks.manager import Callbacks
from langchain_core.prompts.chat import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.tools import BaseTool

from langchain_experimental.pydantic_v1 import BaseModel

DEMONSTRATIONS = [
    {
        "role": "user",
        "content": "покажи мне видео и изображение на основе текста 'мальчик бежит' и озвучь это",  # noqa: E501
    },
    {
        "role": "assistant",
        "content": '[{{"task": "video_generator", "id": 0, "dep": [-1], "args": {{"prompt": "a boy is running" }}}}, {{"task": "text_reader", "id": 1, "dep": [-1], "args": {{"text": "a boy is running" }}}}, {{"task": "image_generator", "id": 2, "dep": [-1], "args": {{"prompt": "a boy is running" }}}}]',  # noqa: E501
    },
    {
        "role": "user",
        "content": "У тебя есть несколько картинок e1.jpg, e2.png, e3.jpg, помоги мне посчитать количество овец?",  # noqa: E501
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
        system_template = """#1 Этап планирования задач: AI-ассистент может разбить ввод пользователя на несколько задач: [{{"task": задача, "id": id_задачи, "dep": id_зависимой_задачи, "args": {{"input name": текст может содержать <resource-dep_id>}}}}]. Специальный тег "dep_id" относится к сгенерированному тексту/изображению/аудио в зависимой задаче (Пожалуйста, учтите, генерирует ли зависимая задача ресурсы этого типа.) и "dep_id" должен быть в списке "dep". Поле "dep" обозначает id предыдущих обязательных задач, которые генерируют новый ресурс, на котором зависит текущая задача. Задача ДОЛЖНА быть выбрана из следующих инструментов (вместе с описанием инструмента, именем ввода и типом вывода): {tools}. Может быть несколько задач одного типа. Подумай шаг за шагом обо всех задачах, необходимых для решения запроса пользователя. Выделите как можно меньше задач, обеспечивая при этом возможность решения запроса пользователя. Обратите внимание на зависимости и порядок задач. Если ввод пользователя не может быть разобран, вам нужно ответить пустым JSON []."""  # noqa: E501
        human_template = """Теперь я ввожу: {input}."""
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
    """A step in the plan."""

    def __init__(
        self, task: str, id: int, dep: List[int], args: Dict[str, str], tool: BaseTool
    ):
        self.task = task
        self.id = id
        self.dep = dep
        self.args = args
        self.tool = tool


class Plan:
    """A plan to execute."""

    def __init__(self, steps: List[Step]):
        self.steps = steps

    def __str__(self) -> str:
        return str([str(step) for step in self.steps])

    def __repr__(self) -> str:
        return str(self)


class BasePlanner(BaseModel):
    """Base class for a planner."""

    @abstractmethod
    def plan(self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any) -> Plan:
        """Given input, decide what to do."""

    @abstractmethod
    async def aplan(
        self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any
    ) -> Plan:
        """Asynchronous Given input, decide what to do."""


class PlanningOutputParser(BaseModel):
    """Parses the output of the planning stage."""

    def parse(self, text: str, hf_tools: List[BaseTool]) -> Plan:
        """Parse the output of the planning stage.

        Args:
            text: The output of the planning stage.
            hf_tools: The tools available.

        Returns:
            The plan.
        """
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
    """Planner for tasks."""

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
        """Asynchronous Given input, decided what to do."""
        inputs["hf_tools"] = [
            f"{tool.name}: {tool.description}" for tool in inputs["hf_tools"]
        ]
        llm_response = await self.llm_chain.arun(
            **inputs, stop=self.stop, callbacks=callbacks
        )
        return self.output_parser.parse(llm_response, inputs["hf_tools"])


def load_chat_planner(llm: BaseLanguageModel) -> TaskPlanner:
    """Load the chat planner."""

    llm_chain = TaskPlaningChain.from_llm(llm)
    return TaskPlanner(llm_chain=llm_chain, output_parser=PlanningOutputParser())
