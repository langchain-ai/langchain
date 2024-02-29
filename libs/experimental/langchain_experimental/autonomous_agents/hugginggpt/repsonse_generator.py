from typing import Any, List, Optional

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import Callbacks
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


class ResponseGenerationChain(LLMChain):
    """Chain to execute tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, verbose: bool = True) -> LLMChain:
        execution_template = (
            "AI-ассистент проанализировал ввод пользователя,"
            " разбил его на несколько задач"
            "и выполнил их. Результаты следующие:\n"
            "{task_execution}"
            "\nПожалуйста, суммируй результаты и сформулируй ответ."
        )
        prompt = PromptTemplate(
            template=execution_template,
            input_variables=["task_execution"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


class ResponseGenerator:
    """Generates a response based on the input."""

    def __init__(self, llm_chain: LLMChain, stop: Optional[List] = None):
        self.llm_chain = llm_chain
        self.stop = stop

    def generate(self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any) -> str:
        """Given input, decided what to do."""
        llm_response = self.llm_chain.run(**inputs, stop=self.stop, callbacks=callbacks)
        return llm_response


def load_response_generator(llm: BaseLanguageModel) -> ResponseGenerator:
    """Load the ResponseGenerator."""

    llm_chain = ResponseGenerationChain.from_llm(llm)
    return ResponseGenerator(
        llm_chain=llm_chain,
    )
