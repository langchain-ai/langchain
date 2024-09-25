"""Chain that interprets a prompt and executes python code to do symbolic math."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from langchain.base_language import BaseLanguageModel
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.prompts.base import BasePromptTemplate
from pydantic import ConfigDict

from langchain_experimental.llm_symbolic_math.prompt import PROMPT


class LLMSymbolicMathChain(Chain):
    """Chain that interprets a prompt and executes python code to do symbolic math.

    It is based on the sympy library and can be used to evaluate
    mathematical expressions.
    See https://www.sympy.org/ for more information.

    Example:
        .. code-block:: python

            from langchain.chains import LLMSymbolicMathChain
            from langchain_community.llms import OpenAI
            llm_symbolic_math = LLMSymbolicMathChain.from_llm(OpenAI())
    """

    llm_chain: LLMChain
    input_key: str = "question"  #: :meta private:
    output_key: str = "answer"  #: :meta private:

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    allow_dangerous_requests: bool  # Assign no default.
    """Must be set by the user to allow dangerous requests or not.

    We recommend a default of False to allow only pre-defined symbolic operations.

    When set to True, the chain will allow any kind of input. This is 
    STRONGLY DISCOURAGED unless you fully trust the input (and believe that 
    the LLM itself cannot behave in a malicious way).
    You should absolutely NOT be deploying this in a production environment
    with allow_dangerous_requests=True. As this would allow a malicious actor
    to execute arbitrary code on your system.
    Use default=True at your own risk.


    When set to False, the chain will only allow pre-defined symbolic operations.
    If the some symbolic expressions are failing to evaluate, you can open a PR
    to add them to extend the list of allowed operations.
    """

    def __init__(self, **kwargs: Any) -> None:
        if "allow_dangerous_requests" not in kwargs:
            raise ValueError(
                "LLMSymbolicMathChain requires allow_dangerous_requests to be set. "
                "We recommend that you set `allow_dangerous_requests=False` to allow "
                "only pre-defined symbolic operations. "
                "If the some symbolic expressions are failing to evaluate, you can "
                "open a PR to add them to extend the list of allowed operations. "
                "Alternatively, you can set `allow_dangerous_requests=True` to allow "
                "any kind of input but this is STRONGLY DISCOURAGED unless you "
                "fully trust the input (and believe that the LLM itself cannot behave "
                "in a malicious way)."
                "You should absolutely NOT be deploying this in a production "
                "environment with allow_dangerous_requests=True. As "
                "this would allow a malicious actor to execute arbitrary code on "
                "your system."
            )
        super().__init__(**kwargs)

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
            import sympy
        except ImportError as e:
            raise ImportError(
                "Unable to import sympy, please install it with `pip install sympy`."
            ) from e

        try:
            if self.allow_dangerous_requests:
                output = str(sympy.sympify(expression, evaluate=True))
            else:
                allowed_symbols = {
                    # Basic arithmetic and trigonometry
                    "sin": sympy.sin,
                    "cos": sympy.cos,
                    "tan": sympy.tan,
                    "cot": sympy.cot,
                    "sec": sympy.sec,
                    "csc": sympy.csc,
                    "asin": sympy.asin,
                    "acos": sympy.acos,
                    "atan": sympy.atan,
                    # Hyperbolic functions
                    "sinh": sympy.sinh,
                    "cosh": sympy.cosh,
                    "tanh": sympy.tanh,
                    "asinh": sympy.asinh,
                    "acosh": sympy.acosh,
                    "atanh": sympy.atanh,
                    # Exponentials and logarithms
                    "exp": sympy.exp,
                    "log": sympy.log,
                    "ln": sympy.log,  # natural log sympy defaults to natural log
                    "log10": lambda x: sympy.log(x, 10),  # log base 10 (use sympy.log)
                    # Powers and roots
                    "sqrt": sympy.sqrt,
                    "cbrt": lambda x: sympy.Pow(x, sympy.Rational(1, 3)),
                    # Combinatorics and other math functions
                    "factorial": sympy.factorial,
                    "binomial": sympy.binomial,
                    "gcd": sympy.gcd,
                    "lcm": sympy.lcm,
                    "abs": sympy.Abs,
                    "sign": sympy.sign,
                    "mod": sympy.Mod,
                    # Constants
                    "pi": sympy.pi,
                    "e": sympy.E,
                    "I": sympy.I,
                    "oo": sympy.oo,
                    "NaN": sympy.nan,
                }

                # Use parse_expr with strict settings
                output = str(
                    sympy.parse_expr(
                        expression, local_dict=allowed_symbols, evaluate=True
                    )
                )
        except Exception as e:
            raise ValueError(
                f'LLMSymbolicMathChain._evaluate("{expression}") raised error: {e}.'
                " Please try again with a valid numerical expression"
            )

        # Remove any leading and trailing brackets from the output
        return re.sub(r"^\[|\]$", "", output)

    def _process_llm_result(
        self, llm_output: str, run_manager: CallbackManagerForChainRun
    ) -> Dict[str, str]:
        run_manager.on_text(llm_output, color="green", verbose=self.verbose)
        llm_output = llm_output.strip()
        text_match = re.search(r"^```text(.*?)```", llm_output, re.DOTALL)
        if text_match:
            expression = text_match.group(1)
            output = self._evaluate_expression(expression)
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
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> Dict[str, str]:
        await run_manager.on_text(llm_output, color="green", verbose=self.verbose)
        llm_output = llm_output.strip()
        text_match = re.search(r"^```text(.*?)```", llm_output, re.DOTALL)
        if text_match:
            expression = text_match.group(1)
            output = self._evaluate_expression(expression)
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
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        _run_manager.on_text(inputs[self.input_key])
        llm_output = self.llm_chain.predict(
            question=inputs[self.input_key],
            stop=["```output"],
            callbacks=_run_manager.get_child(),
        )
        return self._process_llm_result(llm_output, _run_manager)

    async def _acall(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        await _run_manager.on_text(inputs[self.input_key])
        llm_output = await self.llm_chain.apredict(
            question=inputs[self.input_key],
            stop=["```output"],
            callbacks=_run_manager.get_child(),
        )
        return await self._aprocess_llm_result(llm_output, _run_manager)

    @property
    def _chain_type(self) -> str:
        return "llm_symbolic_math_chain"

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: BasePromptTemplate = PROMPT,
        **kwargs: Any,
    ) -> LLMSymbolicMathChain:
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return cls(llm_chain=llm_chain, **kwargs)
