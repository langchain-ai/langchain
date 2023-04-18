"""Chain that interprets a prompt and executes python code to do math."""
import math
import re
from typing import Dict, List

import numexpr
from pydantic import Extra

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.llm_math.prompt import PROMPT
from langchain.prompts.base import BasePromptTemplate
from langchain.schema import BaseLanguageModel


class LLMMathChain(Chain):
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

    def _evaluate_expression(self, expression: str) -> str:
        try:
            local_dict = {"pi": math.pi, "e": math.e}
            output = str(
                numexpr.evaluate(
                    expression.strip(),
                    global_dict={},  # restrict access to globals
                    local_dict=local_dict,  # add common mathematical functions
                )
            )
        except Exception as e:
            raise ValueError(f"{e}. Please try again with a valid numerical expression")

        # Remove any leading and trailing brackets from the output
        return re.sub(r"^\[|\]$", "", output)

    def _process_llm_result(self, llm_output: str) -> Dict[str, str]:
        self.callback_manager.on_text(llm_output, color="green", verbose=self.verbose)
        llm_output = llm_output.strip()
        text_match = re.search(r"^```text(.*?)```", llm_output, re.DOTALL)
        if text_match:
            expression = text_match.group(1)
            output = self._evaluate_expression(expression)
            self.callback_manager.on_text("\nAnswer: ", verbose=self.verbose)
            self.callback_manager.on_text(output, color="yellow", verbose=self.verbose)
            answer = "Answer: " + output
        elif llm_output.startswith("Answer:"):
            answer = llm_output
        elif "Answer:" in llm_output:
            answer = "Answer: " + llm_output.split("Answer:")[-1]
        else:
            raise ValueError(f"unknown format from LLM: {llm_output}")
        return {self.output_key: answer}

    async def _aprocess_llm_result(self, llm_output: str) -> Dict[str, str]:
        if self.callback_manager.is_async:
            await self.callback_manager.on_text(
                llm_output, color="green", verbose=self.verbose
            )
        else:
            self.callback_manager.on_text(
                llm_output, color="green", verbose=self.verbose
            )
        llm_output = llm_output.strip()
        text_match = re.search(r"^```text(.*?)```", llm_output, re.DOTALL)
        if text_match:
            expression = text_match.group(1)
            output = self._evaluate_expression(expression)
            if self.callback_manager.is_async:
                await self.callback_manager.on_text("\nAnswer: ", verbose=self.verbose)
                await self.callback_manager.on_text(
                    output, color="yellow", verbose=self.verbose
                )
            else:
                await self.callback_manager.on_text("\nAnswer: ", verbose=self.verbose)
                await self.callback_manager.on_text(
                    output, color="yellow", verbose=self.verbose
                )
            answer = "Answer: " + output
        elif llm_output.startswith("Answer:"):
            answer = llm_output
        elif "Answer:" in llm_output:
            answer = "Answer: " + llm_output.split("Answer:")[-1]
        else:
            raise ValueError(f"unknown format from LLM: {llm_output}")
        return {self.output_key: answer}

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        llm_executor = LLMChain(
            prompt=self.prompt, llm=self.llm, callback_manager=self.callback_manager
        )
        self.callback_manager.on_text(inputs[self.input_key], verbose=self.verbose)
        llm_output = llm_executor.predict(
            question=inputs[self.input_key], stop=["```output"]
        )
        return self._process_llm_result(llm_output)

    async def _acall(self, inputs: Dict[str, str]) -> Dict[str, str]:
        llm_executor = LLMChain(
            prompt=self.prompt, llm=self.llm, callback_manager=self.callback_manager
        )
        if self.callback_manager.is_async:
            await self.callback_manager.on_text(
                inputs[self.input_key], verbose=self.verbose
            )
        else:
            self.callback_manager.on_text(inputs[self.input_key], verbose=self.verbose)
        llm_output = await llm_executor.apredict(
            question=inputs[self.input_key], stop=["```output"]
        )
        return await self._aprocess_llm_result(llm_output)

    @property
    def _chain_type(self) -> str:
        return "llm_math_chain"
