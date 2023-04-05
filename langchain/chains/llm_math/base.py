"""Chain that interprets a prompt and executes python code to do math."""
from typing import Dict, List

from pydantic import BaseModel, Extra

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.llm_math.prompt import PROMPT
from langchain.prompts.base import BasePromptTemplate
from langchain.python import PythonREPL
from langchain.schema import BaseLanguageModel


class LLMMathChain(Chain, BaseModel):
    """Chain that interprets a prompt and executes python code to do math.

    Example:
        .. code-block:: python

            from langchain import LLMMathChain, OpenAI
            llm_math = LLMMathChain(llm=OpenAI())
    """

    llm: BaseLanguageModel
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

    def _process_llm_result(self, t: str) -> Dict[str, str]:
        python_executor = PythonREPL()
        self.callback_manager.on_text(t, color="green", verbose=self.verbose)
        t = t.strip()
        if t.startswith("```python"):
            code = t[9:-4]
            output = python_executor.run(code)
            self.callback_manager.on_text("\nAnswer: ", verbose=self.verbose)
            self.callback_manager.on_text(output, color="yellow", verbose=self.verbose)
            answer = "Answer: " + output
        elif t.startswith("Answer:"):
            answer = t
        elif "Answer:" in t:
            answer = "Answer: " + t.split("Answer:")[-1]
        else:
            raise ValueError(f"unknown format from LLM: {t}")
        return {self.output_key: answer}

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        llm_executor = LLMChain(
            prompt=self.prompt, llm=self.llm, callback_manager=self.callback_manager
        )
        self.callback_manager.on_text(inputs[self.input_key], verbose=self.verbose)
        t = llm_executor.predict(question=inputs[self.input_key], stop=["```output"])
        return self._process_llm_result(t)

    async def _acall(self, inputs: Dict[str, str]) -> Dict[str, str]:
        llm_executor = LLMChain(
            prompt=self.prompt, llm=self.llm, callback_manager=self.callback_manager
        )
        self.callback_manager.on_text(inputs[self.input_key], verbose=self.verbose)
        t = await llm_executor.apredict(
            question=inputs[self.input_key], stop=["```output"]
        )
        return self._process_llm_result(t)

    @property
    def _chain_type(self) -> str:
        return "llm_math_chain"
