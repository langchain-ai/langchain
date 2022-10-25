"""Chain that interprets a prompt and executes python code to do math."""
from typing import Dict, List

from pydantic import BaseModel, Extra

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.llm_math.prompt import PROMPT
from langchain.chains.python import PythonChain
from langchain.llms.base import LLM


class LLMMathChain(Chain, BaseModel):
    """Chain that interprets a prompt and executes python code to do math.

    Example:
        .. code-block:: python

            from langchain import LLMMathChain, OpenAI
            llm_math = LLMMathChain(llm=OpenAI())
    """

    llm: LLM
    """LLM wrapper to use."""
    verbose: bool = False
    """Whether to print out the code that was executed."""
    input_key: str = "question"  #: :meta private:
    output_key: str = "answer"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Expect output key.

        :meta private:
        """
        return [self.output_key]

    def _run(self, inputs: Dict[str, str]) -> Dict[str, str]:
        llm_executor = LLMChain(prompt=PROMPT, llm=self.llm)
        python_executor = PythonChain()
        question = inputs[self.input_key]
        t = llm_executor.predict(question=question, stop=["```output"]).strip()
        if t.startswith("```python"):
            code = t[9:-4]
            if self.verbose:
                print("[DEBUG] evaluating code")
                print(code)
            output = python_executor.run(code)
            answer = "Answer: " + output
        elif t.startswith("Answer:"):
            answer = t
        else:
            raise ValueError(f"unknown format from LLM: {t}")
        return {self.output_key: answer}

    def run(self, question: str) -> str:
        """Understand user question and execute math in Python if necessary.

        Args:
            question: User question that contains a math question to parse and answer.

        Returns:
            The answer to the question.

        Example:
            .. code-block:: python

                answer = llm_math.run("What is one plus one?")
        """
        return self({self.input_key: question})[self.output_key]
