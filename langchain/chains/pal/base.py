"""Implements Program-Aided Language Models.

As in https://arxiv.org/pdf/2211.10435.pdf.
"""
from __future__ import annotations

import warnings
import ast
from typing import Any, Dict, List, Optional

from pydantic import Extra, root_validator

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.pal.colored_object_prompt import COLORED_OBJECT_PROMPT
from langchain.chains.pal.math_prompt import MATH_PROMPT
from langchain.prompts.base import BasePromptTemplate
from langchain.utilities import PythonREPL

DEFAULT_CODE_VALIDATIONS = {'solution_function': 'solution', 'allow_imports': False, 'allow_non_solution_root_scope_expressions': False, 'allow_non_math_operations': True}

class PALChain(Chain):
    """Implements Program-Aided Language Models."""

    llm_chain: LLMChain
    llm: Optional[BaseLanguageModel] = None
    """[Deprecated]"""
    prompt: BasePromptTemplate = MATH_PROMPT
    """[Deprecated]"""
    stop: str = "\n\n"
    get_answer_expr: str = "print(solution())"
    python_globals: Optional[Dict[str, Any]] = None
    python_locals: Optional[Dict[str, Any]] = None
    output_key: str = "result"  #: :meta private:
    return_intermediate_steps: bool = False
    code_validations: Optional[Dict[str, Any]] = DEFAULT_CODE_VALIDATIONS

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def raise_deprecation(cls, values: Dict) -> Dict:
        if "llm" in values:
            warnings.warn(
                "Directly instantiating an PALChain with an llm is deprecated. "
                "Please instantiate with llm_chain argument or using the one of "
                "the class method constructors from_math_prompt, "
                "from_colored_object_prompt."
            )
            if "llm_chain" not in values and values["llm"] is not None:
                values["llm_chain"] = LLMChain(llm=values["llm"], prompt=MATH_PROMPT)
        return values

    @property
    def input_keys(self) -> List[str]:
        """Return the singular input key.

        :meta private:
        """
        return self.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        """Return the singular output key.

        :meta private:
        """
        if not self.return_intermediate_steps:
            return [self.output_key]
        else:
            return [self.output_key, "intermediate_steps"]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        code = self.llm_chain.predict(
            stop=[self.stop], callbacks=_run_manager.get_child(), **inputs
        )
        _run_manager.on_text(code, color="green", end="\n", verbose=self.verbose)
        PALChain.validate_code(code, self.code_validations)
        repl = PythonREPL(_globals=self.python_globals, _locals=self.python_locals)
        res = repl.run(code + f"\n{self.get_answer_expr}")
        output = {self.output_key: res.strip()}
        if self.return_intermediate_steps:
            output["intermediate_steps"] = code
        return output

    @classmethod
    def validate_code(cls, code, code_validations: Dict[str, Any]):
        try:
            code_tree = ast.parse(code)
        except (SyntaxError, UnicodeDecodeError):
            raise ValueError(f"Generated code is not valid python code: {code}")
        except TypeError:
            raise ValueError(f"Generated code is expected to be a string, instead found {type(code)}")
        except OverflowError:
            raise ValueError(f"Generated code too long / complex to be parsed by ast: {code}")
        
        top_level_nodes = list(ast.iter_child_nodes(code_tree))
        if code_validations.get('allow_non_solution_root_scope_expressions') is False and len(top_level_nodes) > 1:
            raise ValueError(f"Generated code has more than 1 root scope expressions: {code}")
        
        solution_func_name = code_validations.get('solution_function')
        if not isinstance(solution_func_name, str):
            raise ValueError(f"solution_function code validation parameter should be str, instead found {type(solution_func_name)}")
        found_solution_func = False
        has_imports = False
        for node in top_level_nodes:
            if isinstance(node, ast.FunctionDef) and node.name == solution_func_name:
                found_solution_func = True
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                has_imports = True

        if not found_solution_func:
            raise ValueError(f"Generated code is missing the solution function: {code}")
        
        if code_validations.get('allow_imports') is False and has_imports:
            raise ValueError(f"Generated code has disallowed imports: {code}")
        
        if code_validations.get('allow_non_math_operations') is False:
            for node in ast.walk(code_tree):
                #if type(node) not in (ast.Assign, ast.FunctionDef, ast., ast.Add)
                pass

    @classmethod
    def from_math_prompt(cls, llm: BaseLanguageModel, **kwargs: Any) -> PALChain:
        """Load PAL from math prompt."""
        llm_chain = LLMChain(llm=llm, prompt=MATH_PROMPT)
        code_validations = DEFAULT_CODE_VALIDATIONS
        disallow_non_math_operations = kwargs.get('disallow_non_math_operations')
        if disallow_non_math_operations is True:
            code_validations.update({'allow_non_math_operations': False})
        return cls(
            llm_chain=llm_chain,
            stop="\n\n",
            get_answer_expr="print(solution())",
            code_validations = code_validations,
            **kwargs,
        )

    @classmethod
    def from_colored_object_prompt(
        cls, llm: BaseLanguageModel, **kwargs: Any
    ) -> PALChain:
        """Load PAL from colored object prompt."""
        llm_chain = LLMChain(llm=llm, prompt=COLORED_OBJECT_PROMPT)
        return cls(
            llm_chain=llm_chain,
            stop="\n\n\n",
            get_answer_expr="print(answer)",
            **kwargs,
        )

    @property
    def _chain_type(self) -> str:
        return "pal_chain"
