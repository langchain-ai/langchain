import re
from typing import Any

from typing_extensions import override

from langchain_classic.evaluation.schema import StringEvaluator


class RegexMatchStringEvaluator(StringEvaluator):
    """Compute a regex match between the prediction and the reference.

    Examples:
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

    def __init__(self, *, flags: int = 0, **_: Any):  # Default is no flags
        """Initialize the RegexMatchStringEvaluator.

        Args:
            flags: Flags to use for the regex match. Defaults to no flags.
        """
        super().__init__()
        self.flags = flags

    @property
    def requires_input(self) -> bool:
        """This evaluator does not require input."""
        return False

    @property
    def requires_reference(self) -> bool:
        """This evaluator requires a reference."""
        return True

    @property
    def input_keys(self) -> list[str]:
        """Get the input keys.

        Returns:
            The input keys.
        """
        return ["reference", "prediction"]

    @property
    def evaluation_name(self) -> str:
        """Get the evaluation name.

        Returns:
            The evaluation name.
        """
        return "regex_match"

    @override
    def _evaluate_strings(  # type: ignore[override]
        self,
        *,
        prediction: str,
        reference: str,
        **kwargs: Any,
    ) -> dict:
        """Evaluate the regex match between the prediction and the reference.

        Args:
            prediction: The prediction string.
            reference: The reference regex pattern.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            The evaluation results containing the score.
        """
        match = re.match(reference, prediction, flags=self.flags)
        return {"score": int(bool(match))}
