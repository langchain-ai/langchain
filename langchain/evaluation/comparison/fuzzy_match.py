from enum import Enum
from typing import Any, Callable, Dict, Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
    Callbacks,
)
from langchain.chains.base import Chain
from langchain.evaluation.schema import PairwiseStringEvaluator


def _load_rapidfuzz() -> Any:
    try:
        import rapidfuzz
    except ImportError:
        raise ImportError(
            "Please install the rapidfuzz library to use the FuzzyMatchStringEvaluator."
        )
    return rapidfuzz.distance


class StringDistance(str, Enum):
    DAMERAU_LEVENSHTEIN = "damerau_levenshtein"
    LEVENSHTEIN = "levenshtein"
    JARO = "jaro"
    JARO_WINKLER = "jaro_winkler"


class FuzzyMatchStringEvaluator(Chain, PairwiseStringEvaluator):
    def __init__(
        self,
        distance: str = StringDistance.DAMERAU_LEVENSHTEIN,
    ) -> None:
        self.metric = self._get_metric(distance)

    @staticmethod
    def _get_metric(distance: str) -> Callable:
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
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> Dict[str, Any]:
        return {"score": self.metric(inputs["prediction"], inputs["prediction_b"])}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> Dict[str, Any]:
        return {"score": self.metric(inputs["prediction"], inputs["prediction_b"])}

    def evaluate_string_pairs(
        self,
        *,
        prediction: str,
        prediction_b: str,
        input: Optional[str] = None,
        reference: Optional[str] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> dict:
        """Evaluate the string distance between two predictions.

        Args:
            prediction (str): The output string from the first model.
            prediction_b (str): The output string from the second model.
            input (str): The input or task string.
            callbacks (Callbacks, optional): The callbacks to use.
            reference (str, optional): The reference string, if any.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            dict: A dictionary containing:
                - reasoning: The reasoning for the preference.
                - value: The preference value, which is either 'A', 'B', or None
                    for no preference.
                - score: The preference score, which is 1 for 'A', 0 for 'B',
                    and 0.5 for None.
        """
        return self(
            inputs={"prediction": prediction, "prediction_b": prediction_b},
            callbacks=callbacks,
        )

    async def aevaluate_string_pairs(
        self,
        *,
        prediction: str,
        prediction_b: str,
        input: Optional[str] = None,
        reference: Optional[str] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> dict:
        """Asynchronously evaluate the string distance between two predictions.

        Args:
            prediction (str): The output string from the first model.
            prediction_b (str): The output string from the second model.
            input (str): The input or task string.
            callbacks (Callbacks, optional): The callbacks to use.
            reference (str, optional): The reference string, if any.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            dict: A dictionary containing:
                - reasoning: The reasoning for the preference.
                - value: The preference value, which is either 'A', 'B', or None
                    for no preference.
                - score: The preference score, which is 1 for 'A', 0 for 'B',
                    and 0.5 for None.
        """
        return await self.acll(
            inputs={"prediction": prediction, "prediction_b": prediction_b},
            callbacks=callbacks,
        )
