"""Chain to parse / validate the LLM output."""
import json
from typing import Any, Callable, Dict

from langchain.chains.llm import LLMChain
from langchain.prompts.prompt import PromptTemplate

_PROMPT_TEMPLATE_CORRECTION = """
Output: {output}
Validator function: {validator_name}
Exception failed with error: {error_message}

---

Please correct the output so it passes the validator function.
Corrected output:
"""

PROMPT_CORRECTION = PromptTemplate(
    input_variables=["output", "validator_name", "error_message"],
    template=_PROMPT_TEMPLATE_CORRECTION,
)


def boolean_validator(output: str) -> bool:
    """Convert string to boolean."""
    if output.lower() in ("yes", "true", "t", "1"):
        return True
    elif output.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise ValueError("could not convert string to boolean: " + output)


def json_validator(output: str) -> Dict[str, Any]:
    """Convert string to json."""
    return json.loads(output)


VALIDATORS = {
    "boolean": boolean_validator,
    "json": json_validator,
}


class LLMChainWithValidator(LLMChain):
    """Chain that accept a validator function.

    It has special parameters to handle errors in the validator function.
    - `correct_on_error`: If True, the chain will use the output of the validator
      function to try to correct the input.
    - `retry`: Number of times to retry the validator function if it fails
      (done after correcting the input if `correct_on_error` is True)
    """

    correct_on_error: bool = False
    retry: int = 0
    validator: Callable

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """
        Apply validator function to the output of the LLM.

        If the validator function fails, it will try to correct the input
        If the validator function fails after correcting the input,
        it will retry the validator function
        """
        for _ in range(1 + self.retry):
            result = super().apply([inputs])[0]
            try:
                result["response"] = self._validate_output(result["response"])
                return result
            except Exception as exc:
                if self.correct_on_error:
                    try:
                        result["response"] = self._correct_on_error(result, exc)
                        return result
                    except Exception:
                        continue
        return {}

    def _validate_output(self, output: str) -> Any:
        output = output.strip()  # remove spaces and newlines
        return self.validator(output)

    def _correct_on_error(self, result: Dict[str, str], e: Exception) -> str:
        """Try to correct the input if the validator function fails."""
        prompt = PROMPT_CORRECTION.format(
            output=result["response"],
            validator_name=self.validator.__name__,
            error_message=str(e),
        )
        response = self.llm.generate(
            prompts=[prompt],
        )
        output = response.generations[0][0].text
        # apply the validator function to the corrected output
        return self._validate_output(output)
