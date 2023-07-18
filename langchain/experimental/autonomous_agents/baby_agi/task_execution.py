from langchain import LLMChain, PromptTemplate
from langchain.schema.language_model import BaseLanguageModel


class TaskExecutionChain(LLMChain):
    """Chain to execute tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        execution_template = (
            "You are an AI who performs one task based on the following objective: "
            "{objective}."
            "Take into account these previously completed tasks: {context}."
            " Your task: {task}. Response:"
        )
        prompt = PromptTemplate(
            template=execution_template,
            input_variables=["objective", "context", "task"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
