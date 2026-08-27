"""String distance evaluators based on the RapidFuzz library."""

from collections.abc import Callable
from enum import Enum
from typing import Any

from langchain_core.callbacks import Callbacks
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.utils import pre_init
from pydantic import Field
from typing_extensions import override

from langchain_classic.chains.base import Chain
from langchain_classic.evaluation.schema import PairwiseStringEvaluator, StringEvaluator
from langchain_classic.schema import RUN_KEY


def _load_rapidfuzz() -> Any:
    """Load the RapidFuzz library.

    Raises:
        ImportError: If the rapidfuzz library is not installed.

    Returns:
        The `rapidfuzz.distance` module.
    """
    try:
        import rapidfuzz
    except ImportError as e:
        msg = (
            "Please install the rapidfuzz library to use the FuzzyMatchStringEvaluator."
            "Please install it with `pip install rapidfuzz`."
        )
        raise ImportError(msg) from e
    return rapidfuzz.distance


class StringDistance(str, Enum):
    """Distance metric to use.

    Attributes:
        `DAMERAU_LEVENSHTEIN`: The Damerau-Levenshtein distance.
        `LEVENSHTEIN`: The Levenshtein distance.
        `JARO`: The Jaro distance.
        `JARO_WINKLER`: The Jaro-Winkler distance.
        `HAMMING`: The Hamming distance.
        `INDEL`: The Indel distance.
    """

    DAMERAU_LEVENSHTEIN = "damerau_levenshtein"
    LEVENSHTEIN = "levenshtein"
    JARO = "jaro"
    JARO_WINKLER = "jaro_winkler"
    HAMMING = "hamming"
    INDEL = "indel"


class _RapidFuzzChainMixin(Chain):
    """Shared methods for the rapidfuzz string distance evaluators."""

    distance: StringDistance = Field(default=StringDistance.JARO_WINKLER)
    normalize_score: bool = Field(default=True)
    """Whether to normalize the score to a value between `0` and `1`.
    Applies only to the Levenshtein and Damerau-Levenshtein distances."""

    @pre_init
    def validate_dependencies(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Validate that the rapidfuzz library is installed.

        Args:
            values: The input values.

        Returns:
            The validated values.
        """
        _load_rapidfuzz()
        return values

    @property
    def output_keys(self) -> list[str]:
        """Get the output keys.

        Returns:
            The output keys.
        """
        return ["score"]

    def _prepare_output(self, result: dict[str, Any]) -> dict[str, Any]:
        """Prepare the output dictionary.

        Args:
            result: The evaluation results.

        Returns:
            The prepared output dictionary.
        """
        result = {"score": result["score"]}
        if RUN_KEY in result:
            result[RUN_KEY] = result[RUN_KEY].dict()
        return result

    @staticmethod
    def _get_metric(distance: str, *, normalize_score: bool = False) -> Callable:
        """Get the distance metric function based on the distance type.

        Args:
            distance: The distance type.
            normalize_score: Whether to normalize the score.

        Returns:
            The distance metric function.

        Raises:
            ValueError: If the distance metric is invalid.
        """
        from rapidfuzz import distance as rf_distance

        module_map: dict[str, Any] = {
            StringDistance.DAMERAU_LEVENSHTEIN: rf_distance.DamerauLevenshtein,
            StringDistance.LEVENSHTEIN: rf_distance.Levenshtein,
            StringDistance.JARO: rf_distance.Jaro,
            StringDistance.JARO_WINKLER: rf_distance.JaroWinkler,
            StringDistance.HAMMING: rf_distance.Hamming,
            StringDistance.INDEL: rf_distance.Indel,
        }
        if distance not in module_map:
            msg = (
                f"Invalid distance metric: {distance}"
                f"\nMust be one of: {list(StringDistance)}"
            )
            raise ValueError(msg)
        module = module_map[distance]
        if normalize_score:
            return module.normalized_distance
        return module.distance

    @property
    def metric(self) -> Callable:
        """Get the distance metric function.

        Returns:
            The distance metric function.
        """
        return _RapidFuzzChainMixin._get_metric(
            self.distance,
            normalize_score=self.normalize_score,
        )

    def compute_metric(self, a: str, b: str) -> float:
        """Compute the distance between two strings.

        Args:
            a: The first string.
            b: The second string.

        Returns:
            The distance between the two strings.
        """
        return self.metric(a, b)


class StringDistanceEvalChain(StringEvaluator, _RapidFuzzChainMixin):
    """Compute string distances between the prediction and the reference.

    Examples:
    ----------
    >>> from langchain_classic.evaluation import StringDistanceEvalChain
    >>> evaluator = StringDistanceEvalChain()
    >>> evaluator.evaluate_strings(
            prediction="Mindy is the CTO",
            reference="Mindy is the CEO",
        )

    Using the `load_evaluator` function:

    >>> from langchain_classic.evaluation import load_evaluator
    >>> evaluator = load_evaluator("string_distance")
    >>> evaluator.evaluate_strings(
            prediction="The answer is three",
            reference="three",
        )
    """

    @property
    def requires_input(self) -> bool:
        """This evaluator does not require input."""
        return False

    @property
    def requires_reference(self) -> bool:
        """This evaluator does not require a reference."""
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
        return f"{self.distance.value}_distance"

    @override
    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        """Compute the string distance between the prediction and the reference.

        Args:
            inputs: The input values.
            run_manager: The callback manager.

        Returns:
            The evaluation results containing the score.
        """
        return {"score": self.compute_metric(inputs["reference"], inputs["prediction"])}

    @override
    async def _acall(
        self,
        inputs: dict[str, Any],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        """Compute the string distance between the prediction and the reference.

        Args:
            inputs: The input values.
            run_manager: The callback manager.

        Returns:
            The evaluation results containing the score.
        """
        return {"score": self.compute_metric(inputs["reference"], inputs["prediction"])}

    @override
    def _evaluate_strings(
        self,
        *,
        prediction: str,
        reference: str | None = None,
        input: str | None = None,
        callbacks: Callbacks = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        include_run_info: bool = False,
        **kwargs: Any,
    ) -> dict:
        """Evaluate the string distance between the prediction and the reference.

        Args:
            prediction: The prediction string.
            reference: The reference string.
            input: The input string.
            callbacks: The callbacks to use.
            tags: The tags to apply.
            metadata: The metadata to use.
            include_run_info: Whether to include run info in the output.
            **kwargs: Additional keyword arguments.

        Returns:
            The evaluation results containing the score.
        """
        result = self(
            inputs={"prediction": prediction, "reference": reference},
            callbacks=callbacks,
            tags=tags,
            metadata=metadata,
            include_run_info=include_run_info,
        )

        return self._prepare_output(result)

    @override
    async def _aevaluate_strings(
        self,
        *,
        prediction: str,
        reference: str | None = None,
        input: str | None = None,
        callbacks: Callbacks = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        include_run_info: bool = False,
        **kwargs: Any,
    ) -> dict:
        """Evaluate the string distance between the prediction and the reference.

        Args:
            prediction: The prediction string.
            reference: The reference string.
            input: The input string.
            callbacks: The callbacks to use.
            tags: The tags to apply.
            metadata: The metadata to apply.
            include_run_info: Whether to include run info in the output.
            **kwargs: Additional keyword arguments.

        Returns:
            The evaluation results containing the score.
        """
        result = await self.acall(
            inputs={"prediction": prediction, "reference": reference},
            callbacks=callbacks,
            tags=tags,
            metadata=metadata,
            include_run_info=include_run_info,
        )
        return self._prepare_output(result)


class PairwiseStringDistanceEvalChain(PairwiseStringEvaluator, _RapidFuzzChainMixin):
    """Compute string edit distances between two predictions."""

    @property
    def input_keys(self) -> list[str]:
        """Get the input keys.

        Returns:
            The input keys.
        """
        return ["prediction", "prediction_b"]

    @property
    def evaluation_name(self) -> str:
        """Get the evaluation name.

        Returns:
            The evaluation name.
        """
        return f"pairwise_{self.distance.value}_distance"

    @override
    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        """Compute the string distance between two predictions.

        Args:
            inputs: The input values.
            run_manager: The callback manager.

        Returns:
            The evaluation results containing the score.
        """
        return {
            "score": self.compute_metric(inputs["prediction"], inputs["prediction_b"]),
        }

    @override
    async def _acall(
        self,
        inputs: dict[str, Any],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        """Asynchronously compute the string distance between two predictions.

        Args:
            inputs: The input values.
            run_manager: The callback manager.

        Returns:
            The evaluation results containing the score.
        """
        return {
            "score": self.compute_metric(inputs["prediction"], inputs["prediction_b"]),
        }

    @override
    def _evaluate_string_pairs(
        self,
        *,
        prediction: str,
        prediction_b: str,
        callbacks: Callbacks = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        include_run_info: bool = False,
        **kwargs: Any,
    ) -> dict:
        """Evaluate the string distance between two predictions.

        Args:
            prediction: The first prediction string.
            prediction_b: The second prediction string.
            callbacks: The callbacks to use.
            tags: The tags to apply.
            metadata: The metadata to use.
            include_run_info: Whether to include run info in the output.
            **kwargs: Additional keyword arguments.

        Returns:
            The evaluation results containing the score.
        """
        result = self(
            inputs={"prediction": prediction, "prediction_b": prediction_b},
            callbacks=callbacks,
            tags=tags,
            metadata=metadata,
            include_run_info=include_run_info,
        )
        return self._prepare_output(result)

    @override
    async def _aevaluate_string_pairs(
        self,
        *,
        prediction: str,
        prediction_b: str,
        callbacks: Callbacks = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        include_run_info: bool = False,
        **kwargs: Any,
    ) -> dict:
        """Asynchronously evaluate the string distance between two predictions.

        Args:
            prediction: The first prediction string.
            prediction_b: The second prediction string.
            callbacks: The callbacks to use.
            tags: The tags to apply.
            metadata: The metadata to use.
            include_run_info: Whether to include run info in the output.
            **kwargs: Additional keyword arguments.

        Returns:
            The evaluation results containing the score.
        """
        result = await self.acall(
            inputs={"prediction": prediction, "prediction_b": prediction_b},
            callbacks=callbacks,
            tags=tags,
            metadata=metadata,
            include_run_info=include_run_info,
        )
        return self._prepare_output(result)
