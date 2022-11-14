"""Chain that interprets a prompt and executes python code to do math."""
from typing import Dict, List

from pydantic import BaseModel, Extra

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.llm_math.prompt import PROMPT
from langchain.chains.python import PythonChain
from langchain.input import ChainedInput
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

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        llm_executor = LLMChain(prompt=PROMPT, llm=self.llm)
        python_executor = PythonChain()
        chained_input = ChainedInput(inputs[self.input_key], verbose=self.verbose)
        t = llm_executor.predict(question=chained_input.input, stop=["```output"])
        chained_input.add(t, color="green")
        t = t.strip()
        if t.startswith("```python"):
            code = t[9:-4]
            output = python_executor.run(code)
            chained_input.add("\nAnswer: ")
            chained_input.add(output, color="yellow")
            answer = "Answer: " + output
        elif t.startswith("Answer:"):
            answer = t
        else:
            raise ValueError(f"unknown format from LLM: {t}")
        return {self.output_key: answer}
