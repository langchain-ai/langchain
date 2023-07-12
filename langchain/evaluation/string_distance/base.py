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
    """Distance metric to use."""

    DAMERAU_LEVENSHTEIN = "damerau_levenshtein"
    LEVENSHTEIN = "levenshtein"
    JARO = "jaro"
    JARO_WINKLER = "jaro_winkler"


class _RapidFuzzChainMixin(Chain):
    """Shared methods for the rapidfuzz string distance evaluators."""

    distance: StringDistance = Field(default=StringDistance.LEVENSHTEIN)

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


class StringDistanceEvalChain(_RapidFuzzChainMixin, StringEvaluator):
    """Compute string distances between the prediction and the reference."""

    @property
    def requires_input(self) -> bool:
        """
        Check if input is required.

        Returns:
            bool: True if input is required, False otherwise.
        """
        return False

    @property
    def requires_reference(self) -> bool:
        """
        Check if reference is required.

        Returns:
            bool: True if reference is required, False otherwise.
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
        return f"{self.distance.value}_distance"

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
        return {"score": self.metric(inputs["reference"], inputs["prediction"])}

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
        return {"score": self.metric(inputs["reference"], inputs["prediction"])}

    def _evaluate_strings(
        self,
        *,
        prediction: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        callbacks: Callbacks = None,
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
        )
        return {"score": result["score"]}

    async def _aevaluate_strings(
        self,
        *,
        prediction: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        callbacks: Callbacks = None,
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
        )
        return {"score": result["score"]}


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
        return {"score": self.metric(inputs["prediction"], inputs["prediction_b"])}

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
        return {"score": self.metric(inputs["prediction"], inputs["prediction_b"])}

    def _evaluate_string_pairs(
        self,
        *,
        prediction: str,
        prediction_b: str,
        callbacks: Callbacks = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
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
        )
        return {"score": result["score"]}

    async def _aevaluate_string_pairs(
        self,
        *,
        prediction: str,
        prediction_b: str,
        callbacks: Callbacks = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
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
        )
        return {"score": result["score"]}
