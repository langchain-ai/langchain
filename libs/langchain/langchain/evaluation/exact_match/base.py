import string
from typing import Any

from typing_extensions import override

from langchain.evaluation.schema import StringEvaluator


class ExactMatchStringEvaluator(StringEvaluator):
    """Compute an exact match between the prediction and the reference.

    Examples:
    ----------
    >>> evaluator = ExactMatchChain()
    >>> evaluator.evaluate_strings(
            prediction="Mindy is the CTO",
            reference="Mindy is the CTO",
        )  # This will return {'score': 1.0}

    >>> evaluator.evaluate_strings(
            prediction="Mindy is the CTO",
            reference="Mindy is the CEO",
        )  # This will return {'score': 0.0}
    """

    def __init__(
        self,
        *,
        ignore_case: bool = False,
        ignore_punctuation: bool = False,
        ignore_numbers: bool = False,
        **_: Any,
    ):
        """Initialize the ExactMatchStringEvaluator.

        Args:
            ignore_case: Whether to ignore case when comparing strings.
                Defaults to False.
            ignore_punctuation: Whether to ignore punctuation when comparing strings.
                Defaults to False.
            ignore_numbers: Whether to ignore numbers when comparing strings.
                Defaults to False.
        """
        super().__init__()
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
        self.ignore_numbers = ignore_numbers

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
            List[str]: The input keys.
        """
        return ["reference", "prediction"]

    @property
    def evaluation_name(self) -> str:
        """Get the evaluation name.

        Returns:
            str: The evaluation name.
        """
        return "exact_match"

    @override
    def _evaluate_strings(  # type: ignore[override]
        self,
        *,
        prediction: str,
        reference: str,
        **kwargs: Any,
    ) -> dict:
        """Evaluate the exact match between the prediction and the reference.

        Args:
            prediction: The prediction string.
            reference: The reference string.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            The evaluation results containing the score.
        """
        if self.ignore_case:
            prediction = prediction.lower()
            reference = reference.lower()
        if self.ignore_punctuation:
            prediction = prediction.translate(str.maketrans("", "", string.punctuation))
            reference = reference.translate(str.maketrans("", "", string.punctuation))
        if self.ignore_numbers:
            prediction = prediction.translate(str.maketrans("", "", string.digits))
            reference = reference.translate(str.maketrans("", "", string.digits))
        return {"score": int(prediction == reference)}
