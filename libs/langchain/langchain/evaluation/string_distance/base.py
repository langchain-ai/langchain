"""String distance evaluators based on the RapidFuzz library."""

from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import Field, root_validator

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
    Callbacks,
)
from langchain.chains.base import Chain
from langchain.evaluation.schema import PairwiseStringEvaluator, StringEvaluator
from langchain.schema import RUN_KEY


def _load_rapidfuzz() -> Any:
    """
    Load the RapidFuzz library.

    Raises:
        ImportError: If the rapidfuzz library is not installed.

    Returns:
        Any: The rapidfuzz.distance module.
    """
    try:
        import rapidfuzz
    except ImportError:
        raise ImportError(
            "Please install the rapidfuzz library to use the FuzzyMatchStringEvaluator."
        )
    return rapidfuzz.distance


class StringDistance(str, Enum):
    """Distance metric to use.

    Attributes:
        DAMERAU_LEVENSHTEIN: The Damerau-Levenshtein distance.
        LEVENSHTEIN: The Levenshtein distance.
        JARO: The Jaro distance.
        JARO_WINKLER: The Jaro-Winkler distance.
    """

    DAMERAU_LEVENSHTEIN = "damerau_levenshtein"
    LEVENSHTEIN = "levenshtein"
    JARO = "jaro"
    JARO_WINKLER = "jaro_winkler"


class _RapidFuzzChainMixin(Chain):
    """Shared methods for the rapidfuzz string distance evaluators."""

    distance: StringDistance = Field(default=StringDistance.JARO_WINKLER)
    normalize_score: bool = Field(default=True)
    """Whether to normalize the score to a value between 0 and 1.
    Applies only to the Levenshtein and Damerau-Levenshtein distances."""

    @root_validator
    def validate_dependencies(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that the rapidfuzz library is installed.

        Args:
            values (Dict[str, Any]): The input values.

        Returns:
            Dict[str, Any]: The validated values.
        """
        _load_rapidfuzz()
        return values

    @property
    def output_keys(self) -> List[str]:
        """
        Get the output keys.

        Returns:
            List[str]: The output keys.
        """
        return ["score"]

    def _prepare_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare the output dictionary.

        Args:
            result (Dict[str, Any]): The evaluation results.

        Returns:
            Dict[str, Any]: The prepared output dictionary.
        """
        result = {"score": result["score"]}
        if RUN_KEY in result:
            result[RUN_KEY] = result[RUN_KEY].dict()
        return result

    @staticmethod
    def _get_metric(distance: str) -> Callable:
        """
        Get the distance metric function based on the distance type.

        Args:
            distance (str): The distance type.

        Returns:
            Callable: The distance metric function.

        Raises:
            ValueError: If the distance metric is invalid.
        """
        rf_distance = _load_rapidfuzz()
        if distance == StringDistance.DAMERAU_LEVENSHTEIN:
            return rf_distance.DamerauLevenshtein.distance
        elif distance == StringDistance.LEVENSHTEIN:
            return rf_distance.Levenshtein.distance
        elif distance == StringDistance.JARO:
            return rf_distance.Jaro.distance
        elif distance == StringDistance.JARO_WINKLER:
            return rf_distance.JaroWinkler.distance
        else:
            raise ValueError(f"Invalid distance metric: {distance}")

    @property
    def metric(self) -> Callable:
        """
        Get the distance metric function.

        Returns:
            Callable: The distance metric function.
        """
        return _RapidFuzzChainMixin._get_metric(self.distance)

    def compute_metric(self, a: str, b: str) -> float:
        """
        Compute the distance between two strings.

        Args:
            a (str): The first string.
            b (str): The second string.

        Returns:
            float: The distance between the two strings.
        """
        score = self.metric(a, b)
        if self.normalize_score and self.distance in (
            StringDistance.DAMERAU_LEVENSHTEIN,
            StringDistance.LEVENSHTEIN,
        ):
            score = score / max(len(a), len(b))
        return score


class StringDistanceEvalChain(_RapidFuzzChainMixin, StringEvaluator):
    """Compute string distances between the prediction and the reference.

    Examples
    ----------

    >>> from langchain.evaluation import StringDistanceEvalChain
    >>> evaluator = StringDistanceEvalChain()
    >>> evaluator.evaluate_strings(
            prediction="Mindy is the CTO",
            reference="Mindy is the CEO",
        )

    Using the `load_evaluator` function:

    >>> from langchain.evaluation import load_evaluator
    >>> evaluator = load_evaluator("string_distance")
    >>> evaluator.evaluate_strings(
            prediction="The answer is three",
            reference="three",
        )
    """

    @property
    def requires_input(self) -> bool:
        """
        This evaluator does not require input.
        """
        return False

    @property
    def requires_reference(self) -> bool:
        """
        This evaluator does not require a reference.
        """
        return True

    @property
    def input_keys(self) -> List[str]:
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
        return f"{self.distance.value}_distance"

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """
        Compute the string distance between the prediction and the reference.

        Args:
            inputs (Dict[str, Any]): The input values.
            run_manager (Optional[CallbackManagerForChainRun]):
                The callback manager.

        Returns:
            Dict[str, Any]: The evaluation results containing the score.
        """
        return {"score": self.compute_metric(inputs["reference"], inputs["prediction"])}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """
        Asynchronously compute the string distance between the prediction
            and the reference.

        Args:
            inputs (Dict[str, Any]): The input values.
            run_manager (Optional[AsyncCallbackManagerForChainRun]:
                The callback manager.

        Returns:
            Dict[str, Any]: The evaluation results containing the score.
        """
        return {"score": self.compute_metric(inputs["reference"], inputs["prediction"])}

    def _evaluate_strings(
        self,
        *,
        prediction: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        callbacks: Callbacks = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        include_run_info: bool = False,
        **kwargs: Any,
    ) -> dict:
        """
        Evaluate the string distance between the prediction and the reference.

        Args:
            prediction (str): The prediction string.
            reference (Optional[str], optional): The reference string.
            input (Optional[str], optional): The input string.
            callbacks (Callbacks, optional): The callbacks to use.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: The evaluation results containing the score.
        """
        result = self(
            inputs={"prediction": prediction, "reference": reference},
            callbacks=callbacks,
            tags=tags,
            metadata=metadata,
            include_run_info=include_run_info,
        )

        return self._prepare_output(result)

    async def _aevaluate_strings(
        self,
        *,
        prediction: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        callbacks: Callbacks = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        include_run_info: bool = False,
        **kwargs: Any,
    ) -> dict:
        """
        Asynchronously evaluate the string distance between the
            prediction and the reference.

        Args:
            prediction (str): The prediction string.
            reference (Optional[str], optional): The reference string.
            input (Optional[str], optional): The input string.
            callbacks (Callbacks, optional): The callbacks to use.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: The evaluation results containing the score.
        """
        result = await self.acall(
            inputs={"prediction": prediction, "reference": reference},
            callbacks=callbacks,
            tags=tags,
            metadata=metadata,
            include_run_info=include_run_info,
        )
        return self._prepare_output(result)


class PairwiseStringDistanceEvalChain(_RapidFuzzChainMixin, PairwiseStringEvaluator):
    """Compute string edit distances between two predictions."""

    @property
    def input_keys(self) -> List[str]:
        """
        Get the input keys.

        Returns:
            List[str]: The input keys.
        """
        return ["prediction", "prediction_b"]

    @property
    def evaluation_name(self) -> str:
        """
        Get the evaluation name.

        Returns:
            str: The evaluation name.
        """
        return f"pairwise_{self.distance.value}_distance"

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """
        Compute the string distance between two predictions.

        Args:
            inputs (Dict[str, Any]): The input values.
            run_manager (CallbackManagerForChainRun , optional):
                The callback manager.

        Returns:
            Dict[str, Any]: The evaluation results containing the score.
        """
        return {
            "score": self.compute_metric(inputs["prediction"], inputs["prediction_b"])
        }

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """
        Asynchronously compute the string distance between two predictions.

        Args:
            inputs (Dict[str, Any]): The input values.
            run_manager (AsyncCallbackManagerForChainRun , optional):
                The callback manager.

        Returns:
            Dict[str, Any]: The evaluation results containing the score.
        """
        return {
            "score": self.compute_metric(inputs["prediction"], inputs["prediction_b"])
        }

    def _evaluate_string_pairs(
        self,
        *,
        prediction: str,
        prediction_b: str,
        callbacks: Callbacks = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        include_run_info: bool = False,
        **kwargs: Any,
    ) -> dict:
        """
        Evaluate the string distance between two predictions.

        Args:
            prediction (str): The first prediction string.
            prediction_b (str): The second prediction string.
            callbacks (Callbacks, optional): The callbacks to use.
            tags (List[str], optional): Tags to apply to traces.
            metadata (Dict[str, Any], optional): Metadata to apply to traces.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: The evaluation results containing the score.
        """
        result = self(
            inputs={"prediction": prediction, "prediction_b": prediction_b},
            callbacks=callbacks,
            tags=tags,
            metadata=metadata,
            include_run_info=include_run_info,
        )
        return self._prepare_output(result)

    async def _aevaluate_string_pairs(
        self,
        *,
        prediction: str,
        prediction_b: str,
        callbacks: Callbacks = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        include_run_info: bool = False,
        **kwargs: Any,
    ) -> dict:
        """
        Asynchronously evaluate the string distance between two predictions.

        Args:
            prediction (str): The first prediction string.
            prediction_b (str): The second prediction string.
            callbacks (Callbacks, optional): The callbacks to use.
            tags (List[str], optional): Tags to apply to traces.
            metadata (Dict[str, Any], optional): Metadata to apply to traces.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: The evaluation results containing the score.
        """
        result = await self.acall(
            inputs={"prediction": prediction, "prediction_b": prediction_b},
            callbacks=callbacks,
            tags=tags,
            metadata=metadata,
            include_run_info=include_run_info,
        )
        return self._prepare_output(result)
