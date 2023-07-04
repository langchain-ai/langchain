from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
    Callbacks,
)
from langchain.chains.base import Chain
from langchain.evaluation.schema import StringEvaluator


def _load_rapidfuzz() -> Any:
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


class FuzzyMatchStringEvaluator(Chain, StringEvaluator):
    """Apply fuzzy matching to a string."""

    distance: Callable = Field(default=StringDistance.LEVENSHTEIN)

    @property
    def requires_input(self) -> bool:
        return False

    @property
    def requires_reference(self) -> bool:
        return True

    @property
    def input_keys(self) -> List[str]:
        return ["reference", "prediction"]

    @property
    def output_keys(self) -> List[str]:
        return ["score"]

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

    @property
    def metric(self) -> Callable:
        return FuzzyMatchStringEvaluator._get_metric(self.distance)

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> Dict[str, Any]:
        return {"score": self.metric(inputs["reference"], inputs["prediction"])}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> Dict[str, Any]:
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
        """Evaluate fuzzy string difference between prediction and reference.

        Args:
            prediction (str): the LLM or chain prediction to evaluate.
            reference (str): the reference label
                to evaluate against.
            input (Optional[str], optional): the input to consider during evaluation
            **kwargs: additional keyword arguments, including callbacks, tags, etc.
        Returns:
            dict: The evaluation results containing the score or value.
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
        result = await self.acall(
            inputs={"prediction": prediction, "reference": reference},
            callbacks=callbacks,
        )
        return {"score": result["score"]}
