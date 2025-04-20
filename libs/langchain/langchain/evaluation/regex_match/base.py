import re
from typing import Any

from langchain.evaluation.schema import StringEvaluator


class RegexMatchStringEvaluator(StringEvaluator):
    """Compute a regex match between the prediction and the reference.

    Examples
    ----------
    >>> evaluator = RegexMatchStringEvaluator(flags=re.IGNORECASE)
    >>> evaluator.evaluate_strings(
            prediction="Mindy is the CTO",
            reference="^mindy.*cto$",
        )  # This will return {'score': 1.0} due to the IGNORECASE flag

    >>> evaluator = RegexMatchStringEvaluator()
    >>> evaluator.evaluate_strings(
            prediction="Mindy is the CTO",
            reference="^Mike.*CEO$",
        )  # This will return {'score': 0.0}

    >>> evaluator.evaluate_strings(
            prediction="Mindy is the CTO",
            reference="^Mike.*CEO$|^Mindy.*CTO$",
        )  # This will return {'score': 1.0} as the prediction matches the second pattern in the union
    """  # noqa: E501

    def __init__(self, *, flags: int = 0, **kwargs: Any):  # Default is no flags
        super().__init__()
        self.flags = flags

    @property
    def requires_input(self) -> bool:
        """
        This evaluator does not require input.
        """
        return False

    @property
    def requires_reference(self) -> bool:
        """
        This evaluator requires a reference.
        """
        return True

    @property
    def input_keys(self) -> list[str]:
        """
        Get the input keys.

        Returns:
            List[str]: The input keys.
        """
        return ["reference", "prediction"]

    @property
    def evaluation_name(self) -> str:
        """
        Get the evaluation name.

        Returns:
            str: The evaluation name.
        """
        return "regex_match"

    def _evaluate_strings(  # type: ignore[override]
        self,
        *,
        prediction: str,
        reference: str,
        **kwargs: Any,
    ) -> dict:
        """
        Evaluate the regex match between the prediction and the reference.

        Args:
            prediction (str): The prediction string.
            reference (Optional[str], optional): The reference regex pattern.

        Returns:
            dict: The evaluation results containing the score.
        """
        match = re.match(reference, prediction, flags=self.flags)
        return {"score": int(bool(match))}
