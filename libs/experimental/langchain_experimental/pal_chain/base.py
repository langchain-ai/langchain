"""Implements Program-Aided Language Models.

This module implements the Program-Aided Language Models (PAL) for generating code
solutions. PAL is a technique described in the paper "Program-Aided Language Models"
(https://arxiv.org/pdf/2211.10435.pdf).
"""

from __future__ import annotations

import ast
from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.schema.language_model import BaseLanguageModel
from langchain.utilities import PythonREPL

from langchain_experimental.pal_chain.colored_object_prompt import COLORED_OBJECT_PROMPT
from langchain_experimental.pal_chain.math_prompt import MATH_PROMPT
from langchain_experimental.pydantic_v1 import Extra, Field

COMMAND_EXECUTION_FUNCTIONS = ["system", "exec", "execfile", "eval"]


class PALValidation:
    SOLUTION_EXPRESSION_TYPE_FUNCTION = ast.FunctionDef
    SOLUTION_EXPRESSION_TYPE_VARIABLE = ast.Name

    def __init__(
        self,
        solution_expression_name: Optional[str] = None,
        solution_expression_type: Optional[type] = None,
        allow_imports: bool = False,
        allow_command_exec: bool = False,
    ):
        """Initialize a PALValidation instance.

        Args:
            solution_expression_name (str): Name of the expected solution expression.
                If passed, solution_expression_type must be passed as well.
            solution_expression_type (str): AST type of the expected solution
                expression. If passed, solution_expression_name must be passed as well.
                Must be one of PALValidation.SOLUTION_EXPRESSION_TYPE_FUNCTION,
                PALValidation.SOLUTION_EXPRESSION_TYPE_VARIABLE.
            allow_imports (bool): Allow import statements.
            allow_command_exec (bool): Allow using known command execution functions.
        """
        self.solution_expression_name = solution_expression_name
        self.solution_expression_type = solution_expression_type

        if solution_expression_name is not None:
            if not isinstance(self.solution_expression_name, str):
                raise ValueError(
                    f"Expected solution_expression_name to be str, "
                    f"instead found {type(self.solution_expression_name)}"
                )
        if solution_expression_type is not None:
            if (
                self.solution_expression_type
                is not self.SOLUTION_EXPRESSION_TYPE_FUNCTION
                and self.solution_expression_type
                is not self.SOLUTION_EXPRESSION_TYPE_VARIABLE
            ):
                raise ValueError(
                    f"Expected solution_expression_type to be one of "
                    f"({self.SOLUTION_EXPRESSION_TYPE_FUNCTION},"
                    f"{self.SOLUTION_EXPRESSION_TYPE_VARIABLE}),"
                    f"instead found {self.solution_expression_type}"
                )

        if solution_expression_name is not None and solution_expression_type is None:
            raise TypeError(
                "solution_expression_name "
                "requires solution_expression_type to be passed as well"
            )
        if solution_expression_name is None and solution_expression_type is not None:
            raise TypeError(
                "solution_expression_type "
                "requires solution_expression_name to be passed as well"
            )

        self.allow_imports = allow_imports
        self.allow_command_exec = allow_command_exec


class PALChain(Chain):
    """Implements Program-Aided Language Models (PAL).

    This class implements the Program-Aided Language Models (PAL) for generating code
    solutions. PAL is a technique described in the paper "Program-Aided Language Models"
    (https://arxiv.org/pdf/2211.10435.pdf).
    """

    llm_chain: LLMChain
    stop: str = "\n\n"
    """Stop token to use when generating code."""
    get_answer_expr: str = "print(solution())"
    """Expression to use to get the answer from the generated code."""
    python_globals: Optional[Dict[str, Any]] = None
    """Python globals and locals to use when executing the generated code."""
    python_locals: Optional[Dict[str, Any]] = None
    """Python globals and locals to use when executing the generated code."""
    output_key: str = "result"  #: :meta private:
    return_intermediate_steps: bool = False
    """Whether to return intermediate steps in the generated code."""
    code_validations: PALValidation = Field(default_factory=PALValidation)
    """Validations to perform on the generated code."""
    timeout: Optional[int] = 10
    """Timeout in seconds for the generated code to execute."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Return the singular input key.

        :meta private:
        """
        return self.llm_chain.prompt.input_variables

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
        res = repl.run(code + f"\n{self.get_answer_expr}", timeout=self.timeout)
        output = {self.output_key: res.strip()}
        if self.return_intermediate_steps:
            output["intermediate_steps"] = code
        return output

    @classmethod
    def validate_code(cls, code: str, code_validations: PALValidation) -> None:
        try:
            code_tree = ast.parse(code)
        except (SyntaxError, UnicodeDecodeError):
            raise ValueError(f"Generated code is not valid python code: {code}")
        except TypeError:
            raise ValueError(
                f"Generated code is expected to be a string, "
                f"instead found {type(code)}"
            )
        except OverflowError:
            raise ValueError(
                f"Generated code too long / complex to be parsed by ast: {code}"
            )

        found_solution_expr = False
        if code_validations.solution_expression_name is None:
            # Skip validation if no solution_expression_name was given
            found_solution_expr = True

        has_imports = False
        top_level_nodes = list(ast.iter_child_nodes(code_tree))
        for node in top_level_nodes:
            if (
                code_validations.solution_expression_name is not None
                and code_validations.solution_expression_type is not None
            ):
                # Check root nodes (like func def)
                if (
                    isinstance(node, code_validations.solution_expression_type)
                    and hasattr(node, "name")
                    and node.name == code_validations.solution_expression_name
                ):
                    found_solution_expr = True
                # Check assigned nodes (like answer variable)
                if isinstance(node, ast.Assign):
                    for target_node in node.targets:
                        if (
                            isinstance(
                                target_node, code_validations.solution_expression_type
                            )
                            and hasattr(target_node, "id")
                            and target_node.id
                            == code_validations.solution_expression_name
                        ):
                            found_solution_expr = True
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                has_imports = True

        if not found_solution_expr:
            raise ValueError(
                f"Generated code is missing the solution expression: "
                f"{code_validations.solution_expression_name} of type: "
                f"{code_validations.solution_expression_type}"
            )

        if not code_validations.allow_imports and has_imports:
            raise ValueError(f"Generated code has disallowed imports: {code}")

        if (
            not code_validations.allow_command_exec
            or not code_validations.allow_imports
        ):
            for node in ast.walk(code_tree):
                if (
                    (not code_validations.allow_command_exec)
                    and isinstance(node, ast.Call)
                    and (
                        (
                            hasattr(node.func, "id")
                            and node.func.id in COMMAND_EXECUTION_FUNCTIONS
                        )
                        or (
                            isinstance(node.func, ast.Attribute)
                            and node.func.attr in COMMAND_EXECUTION_FUNCTIONS
                        )
                    )
                ):
                    raise ValueError(
                        f"Found illegal command execution function "
                        f"{node.func.id} in code {code}"
                    )

                if (not code_validations.allow_imports) and (
                    isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom)
                ):
                    raise ValueError(f"Generated code has disallowed imports: {code}")

    @classmethod
    def from_math_prompt(cls, llm: BaseLanguageModel, **kwargs: Any) -> PALChain:
        """Load PAL from math prompt.

        Args:
            llm (BaseLanguageModel): The language model to use for generating code.

        Returns:
            PALChain: An instance of PALChain.
        """
        llm_chain = LLMChain(llm=llm, prompt=MATH_PROMPT)
        code_validations = PALValidation(
            solution_expression_name="solution",
            solution_expression_type=PALValidation.SOLUTION_EXPRESSION_TYPE_FUNCTION,
        )

        return cls(
            llm_chain=llm_chain,
            stop="\n\n",
            get_answer_expr="print(solution())",
            code_validations=code_validations,
            **kwargs,
        )

    @classmethod
    def from_colored_object_prompt(
        cls, llm: BaseLanguageModel, **kwargs: Any
    ) -> PALChain:
        """Load PAL from colored object prompt.

        Args:
            llm (BaseLanguageModel): The language model to use for generating code.

        Returns:
            PALChain: An instance of PALChain.
        """
        llm_chain = LLMChain(llm=llm, prompt=COLORED_OBJECT_PROMPT)
        code_validations = PALValidation(
            solution_expression_name="answer",
            solution_expression_type=PALValidation.SOLUTION_EXPRESSION_TYPE_VARIABLE,
        )
        return cls(
            llm_chain=llm_chain,
            stop="\n\n\n",
            get_answer_expr="print(answer)",
            code_validations=code_validations,
            **kwargs,
        )

    @property
    def _chain_type(self) -> str:
        return "pal_chain"
