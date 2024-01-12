from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel


class TaskPrioritizationChain(LLMChain):
    """Chain to prioritize tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_prioritization_template = (
            "Ты - AI, задача которого - привести в порядок"
            " и переприоритизировать следующие задачи: {task_names}."
            " Учти конечную цель твоей команды: {objective}."
            " Не удаляй ни одну из задач. Верни"
            " результат в виде нумерованного списка, например:"
            " #. Первая задача"
            " #. Вторая задача"
            " Начни список задач с номера {next_task_id}."
        )
        prompt = PromptTemplate(
            template=task_prioritization_template,
            input_variables=["task_names", "next_task_id", "objective"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
