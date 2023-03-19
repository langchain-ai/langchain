from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

from pydantic import BaseModel


class GuardrailEvaluation(BaseModel):
    """Hm want to encapsulate the result of applying a guardrail
    """
    # It may fail.
    error_msg: str
    # Optionally, it may retry upon failure. The retry can also fail.
    revised_output: Any


class Guardrail(ABC, BaseModel):

    @abstractmethod
    def evaluate(self, input: Any, output: Any) -> Tuple[Optional[GuardrailEvaluation], bool]:
        """A generic guardrail on any function (a function that gets human input, an LM call, a chain, an agent, etc.)
        is evaluated against that function's input and output.

        Evaluation includes a validation/verification step. It may also include a retry to generate a satisfactory revised output.
        """
