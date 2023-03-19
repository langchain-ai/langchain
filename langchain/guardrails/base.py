from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

from pydantic import BaseModel


class GuardrailEvaluation(BaseModel):
    """Hm want to encapsulate the result of applying a guardrail
    """
    error_msg: str                  # Indicate why initial output validation failed.
    revised_output: Optional[Any]   # Optionally, try to fix the output.


class Guardrail(ABC, BaseModel):

    @abstractmethod
    def evaluate(self, input: Any, output: Any) -> Tuple[Optional[GuardrailEvaluation], bool]:
        """A generic guardrail on any function (a function that gets human input, an LM call, a chain, an agent, etc.)
        is evaluated against that function's input and output.

        Evaluation includes a validation/verification step. It may also include a retry to generate a satisfactory revised output.
        These steps are encapsulated jointly, as a single LM call may succeed in both.
        """
