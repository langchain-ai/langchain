"""Chain that interprets a prompt and executes python code to do math."""
import math
import re
from typing import Dict, List, Optional

import numexpr
from pydantic import Extra

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.llm_math.prompt import PROMPT
from langchain.prompts.base import BasePromptTemplate


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

    def _process_llm_result(
        self, llm_output: str, run_manager: Optional[CallbackManagerForChainRun] = None
    ) -> Dict[str, str]:
        if run_manager:
            run_manager.on_text(llm_output, color="green", verbose=self.verbose)
        llm_output = llm_output.strip()
        text_match = re.search(r"^```text(.*?)```", llm_output, re.DOTALL)
        if text_match:
            expression = text_match.group(1)
            output = self._evaluate_expression(expression)
            if run_manager:
                run_manager.on_text("\nAnswer: ", verbose=self.verbose)
                run_manager.on_text(output, color="yellow", verbose=self.verbose)
            answer = "Answer: " + output
        elif llm_output.startswith("Answer:"):
            answer = llm_output
        elif "Answer:" in llm_output:
            answer = "Answer: " + llm_output.split("Answer:")[-1]
        else:
            raise ValueError(f"unknown format from LLM: {llm_output}")
        return {self.output_key: answer}

    async def _aprocess_llm_result(
        self,
        llm_output: str,
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        if run_manager:
            await run_manager.on_text(llm_output, color="green", verbose=self.verbose)
        llm_output = llm_output.strip()
        text_match = re.search(r"^```text(.*?)```", llm_output, re.DOTALL)
        if text_match:
            expression = text_match.group(1)
            output = self._evaluate_expression(expression)
            if run_manager:
                await run_manager.on_text("\nAnswer: ", verbose=self.verbose)
                await run_manager.on_text(output, color="yellow", verbose=self.verbose)
            answer = "Answer: " + output
        elif llm_output.startswith("Answer:"):
            answer = llm_output
        elif "Answer:" in llm_output:
            answer = "Answer: " + llm_output.split("Answer:")[-1]
        else:
            raise ValueError(f"unknown format from LLM: {llm_output}")
        return {self.output_key: answer}

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        llm_executor = LLMChain(prompt=self.prompt, llm=self.llm)
        if run_manager:
            run_manager.on_text(inputs[self.input_key])
        llm_output = llm_executor.predict(
            question=inputs[self.input_key],
            stop=["```output"],
            callbacks=run_manager.get_child() if run_manager else None,
        )
        return self._process_llm_result(llm_output, run_manager=run_manager)

    async def _acall(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        llm_executor = LLMChain(prompt=self.prompt, llm=self.llm)
        if run_manager:
            await run_manager.on_text(inputs[self.input_key])
        llm_output = await llm_executor.apredict(
            question=inputs[self.input_key],
            stop=["```output"],
            callbacks=run_manager.get_child() if run_manager else None,
        )
        return await self._aprocess_llm_result(llm_output, run_manager=run_manager)

    @property
    def _chain_type(self) -> str:
        return "llm_math_chain"
