from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel


class TaskCreationChain(LLMChain):
    """Chain generating tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_creation_template = (
            "Ты - AI, создающий задачи,"
            " который использует результат работы исполнительного агента"
            " для создания новых задач с следующей целью: {objective},"
            " Последняя выполненная задача имеет результат: {result}."
            " Этот результат был основан на этом описании задачи: {task_description}."
            " Вот незавершенные задачи: {incomplete_tasks}."
            " Основываясь на результате, создай новые задачи для выполнения"
            " AI системой, которые не пересекаются с незавершенными задачами."
            " Верни задачи в виде массива."
        )
        prompt = PromptTemplate(
            template=task_creation_template,
            input_variables=[
                "result",
                "task_description",
                "incomplete_tasks",
                "objective",
            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
