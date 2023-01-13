"""Chain that interprets a prompt and executes python code to do math."""
from typing import Dict, List

from pydantic import BaseModel, Extra

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.llm_math.prompt import PROMPT
from langchain.llms.base import BaseLLM
from langchain.prompts.base import BasePromptTemplate
from langchain.python import PythonREPL


class LLMMathChain(Chain, BaseModel):
    """Chain that interprets a prompt and executes python code to do math.

    Example:
        .. code-block:: python

            from langchain import LLMMathChain, OpenAI
            llm_math = LLMMathChain(llm=OpenAI())
    """

    llm: BaseLLM
    """LLM wrapper to use."""
    prompt: BasePromptTemplate = PROMPT
    """Prompt to use to translate to python if neccessary."""
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
        llm_executor = LLMChain(prompt=self.prompt, llm=self.llm)
        python_executor = PythonREPL()
        if self.verbose:
            self.callback_manager.on_text(inputs[self.input_key])
        t = llm_executor.predict(question=inputs[self.input_key], stop=["```output"])
        if self.verbose:
            self.callback_manager.on_text(t, color="green")
        t = t.strip()
        if t.startswith("```python"):
            code = t[9:-4]
            output = python_executor.run(code)
            if self.verbose:
                self.callback_manager.on_text("\nAnswer: ")
                self.callback_manager.on_text(output, color="yellow")
            answer = "Answer: " + output
        elif t.startswith("Answer:"):
            answer = t
        else:
            raise ValueError(f"unknown format from LLM: {t}")
        return {self.output_key: answer}
