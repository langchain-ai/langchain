from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel


class TaskExecutionChain(LLMChain):
    """Chain to execute tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        execution_template = (
            "Ты - искусственный интеллект, который"
            " выполняет одну задачу на основе следующей цели: "
            "{objective}."
            "Учти эти ранее выполненные задачи: {context}."
            " Твоя задача: {task}. Ответ:"
        )
        prompt = PromptTemplate(
            template=execution_template,
            input_variables=["objective", "context", "task"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
