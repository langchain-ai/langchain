"""A chain for comparing the output of two models using embeddings."""
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
    Callbacks,
)
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import Field, root_validator

from langchain.chains.base import Chain
from langchain.evaluation.schema import PairwiseStringEvaluator, StringEvaluator
from langchain.schema import RUN_KEY
from langchain.utils.math import cosine_similarity


def _embedding_factory() -> Embeddings:
    """Create an Embeddings object.
    Returns:
        Embeddings: The created Embeddings object.
    """
    # Here for backwards compatibility.
    # Generally, we do not want to be seeing imports from langchain community
    # or partner packages in langchain.
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        try:
            from langchain_community.embeddings.openai import OpenAIEmbeddings
        except ImportError:
            raise ImportError(
                "Could not import OpenAIEmbeddings. Please install the "
                "OpenAIEmbeddings package using `pip install langchain-openai`."
            )
    return OpenAIEmbeddings()


class EmbeddingDistance(str, Enum):
    """Embedding Distance Metric.

    Attributes:
        COSINE: Cosine distance metric.
        EUCLIDEAN: Euclidean distance metric.
        MANHATTAN: Manhattan distance metric.
        CHEBYSHEV: Chebyshev distance metric.
        HAMMING: Hamming distance metric.
    """

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    CHEBYSHEV = "chebyshev"
    HAMMING = "hamming"


class _EmbeddingDistanceChainMixin(Chain):
    """Shared functionality for embedding distance evaluators.

    Attributes:
        embeddings (Embeddings): The embedding objects to vectorize the outputs.
        distance_metric (EmbeddingDistance): The distance metric to use
                                            for comparing the embeddings.
    """

    embeddings: Embeddings = Field(default_factory=_embedding_factory)
    distance_metric: EmbeddingDistance = Field(default=EmbeddingDistance.COSINE)

    @root_validator(pre=False)
    def _validate_tiktoken_installed(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that the TikTok library is installed.

        Args:
            values (Dict[str, Any]): The values to validate.

        Returns:
            Dict[str, Any]: The validated values.
        """
        embeddings = values.get("embeddings")
        types_ = []
        try:
            from langchain_openai import OpenAIEmbeddings

            types_.append(OpenAIEmbeddings)
        except ImportError:
            pass

        try:
            from langchain_community.embeddings.openai import OpenAIEmbeddings

            types_.append(OpenAIEmbeddings)
        except ImportError:
            pass

        if not types_:
            raise ImportError(
                "Could not import OpenAIEmbeddings. Please install the "
                "OpenAIEmbeddings package using `pip install langchain-openai`."
            )

        if isinstance(embeddings, tuple(types_)):
            try:
                import tiktoken  # noqa: F401
            except ImportError:
                raise ImportError(
                    "The tiktoken library is required to use the default "
                    "OpenAI embeddings with embedding distance evaluators."
                    " Please either manually select a different Embeddings object"
                    " or install tiktoken using `pip install tiktoken`."
                )
        return values

    class Config:
        """Permit embeddings to go unvalidated."""

        arbitrary_types_allowed: bool = True

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys of the chain.

        Returns:
            List[str]: The output keys.
        """
        return ["score"]

    def _prepare_output(self, result: dict) -> dict:
        parsed = {"score": result["score"]}
        if RUN_KEY in result:
            parsed[RUN_KEY] = result[RUN_KEY]
        return parsed

    def _get_metric(self, metric: EmbeddingDistance) -> Any:
        """Get the metric function for the given metric name.

        Args:
            metric (EmbeddingDistance): The metric name.

        Returns:
            Any: The metric function.
        """
        metrics = {
            EmbeddingDistance.COSINE: self._cosine_distance,
            EmbeddingDistance.EUCLIDEAN: self._euclidean_distance,
            EmbeddingDistance.MANHATTAN: self._manhattan_distance,
            EmbeddingDistance.CHEBYSHEV: self._chebyshev_distance,
            EmbeddingDistance.HAMMING: self._hamming_distance,
        }
        if metric in metrics:
            return metrics[metric]
        else:
            raise ValueError(f"Invalid metric: {metric}")

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute the cosine distance between two vectors.

        Args:
            a (np.ndarray): The first vector.
            b (np.ndarray): The second vector.

        Returns:
            np.ndarray: The cosine distance.
        """
        return 1.0 - cosine_similarity(a, b)

    @staticmethod
    def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.floating:
        """Compute the Euclidean distance between two vectors.

        Args:
            a (np.ndarray): The first vector.
            b (np.ndarray): The second vector.

        Returns:
            np.floating: The Euclidean distance.
        """
        return np.linalg.norm(a - b)

    @staticmethod
    def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> np.floating:
        """Compute the Manhattan distance between two vectors.

        Args:
            a (np.ndarray): The first vector.
            b (np.ndarray): The second vector.

        Returns:
            np.floating: The Manhattan distance.
        """
        return np.sum(np.abs(a - b))

    @staticmethod
    def _chebyshev_distance(a: np.ndarray, b: np.ndarray) -> np.floating:
        """Compute the Chebyshev distance between two vectors.

        Args:
            a (np.ndarray): The first vector.
            b (np.ndarray): The second vector.

        Returns:
            np.floating: The Chebyshev distance.
        """
        return np.max(np.abs(a - b))

    @staticmethod
    def _hamming_distance(a: np.ndarray, b: np.ndarray) -> np.floating:
        """Compute the Hamming distance between two vectors.

        Args:
            a (np.ndarray): The first vector.
            b (np.ndarray): The second vector.

        Returns:
            np.floating: The Hamming distance.
        """
        return np.mean(a != b)

    def _compute_score(self, vectors: np.ndarray) -> float:
        """Compute the score based on the distance metric.

        Args:
            vectors (np.ndarray): The input vectors.

        Returns:
            float: The computed score.
        """
        metric = self._get_metric(self.distance_metric)
        score = metric(vectors[0].reshape(1, -1), vectors[1].reshape(1, -1)).item()
        return score


class EmbeddingDistanceEvalChain(_EmbeddingDistanceChainMixin, StringEvaluator):
    """Use embedding distances to score semantic difference between
    a prediction and reference.

    Examples:
        >>> chain = EmbeddingDistanceEvalChain()
        >>> result = chain.evaluate_strings(prediction="Hello", reference="Hi")
        >>> print(result)
        {'score': 0.5}
    """

    @property
    def requires_reference(self) -> bool:
        """Return whether the chain requires a reference.

        Returns:
            bool: True if a reference is required, False otherwise.
        """
        return True

    @property
    def evaluation_name(self) -> str:
        return f"embedding_{self.distance_metric.value}_distance"

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys of the chain.

        Returns:
            List[str]: The input keys.
        """
        return ["prediction", "reference"]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Compute the score for a prediction and reference.

        Args:
            inputs (Dict[str, Any]): The input data.
            run_manager (Optional[CallbackManagerForChainRun], optional):
                The callback manager.

        Returns:
            Dict[str, Any]: The computed score.
        """
        vectors = np.array(
            self.embeddings.embed_documents([inputs["prediction"], inputs["reference"]])
        )
        score = self._compute_score(vectors)
        return {"score": score}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Asynchronously compute the score for a prediction and reference.

        Args:
            inputs (Dict[str, Any]): The input data.
            run_manager (AsyncCallbackManagerForChainRun, optional):
                The callback manager.

        Returns:
            Dict[str, Any]: The computed score.
        """
        embedded = await self.embeddings.aembed_documents(
            [inputs["prediction"], inputs["reference"]]
        )
        vectors = np.array(embedded)
        score = self._compute_score(vectors)
        return {"score": score}

    def _evaluate_strings(
        self,
        *,
        prediction: str,
        reference: Optional[str] = None,
        callbacks: Callbacks = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        include_run_info: bool = False,
        **kwargs: Any,
    ) -> dict:
        """Evaluate the embedding distance between a prediction and
        reference.

        Args:
            prediction (str): The output string from the first model.
            reference (str): The reference string (required)
            callbacks (Callbacks, optional): The callbacks to use.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            dict: A dictionary containing:
                - score: The embedding distance between the two
                    predictions.
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
        callbacks: Callbacks = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        include_run_info: bool = False,
        **kwargs: Any,
    ) -> dict:
        """Asynchronously evaluate the embedding distance between
        a prediction and reference.

        Args:
            prediction (str): The output string from the first model.
            reference (str): The output string from the second model.
            callbacks (Callbacks, optional): The callbacks to use.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            dict: A dictionary containing:
                - score: The embedding distance between the two
                    predictions.
        """
        result = await self.acall(
            inputs={"prediction": prediction, "reference": reference},
            callbacks=callbacks,
            tags=tags,
            metadata=metadata,
            include_run_info=include_run_info,
        )
        return self._prepare_output(result)


class PairwiseEmbeddingDistanceEvalChain(
    _EmbeddingDistanceChainMixin, PairwiseStringEvaluator
):
    """Use embedding distances to score semantic difference between two predictions.

    Examples:
    >>> chain = PairwiseEmbeddingDistanceEvalChain()
    >>> result = chain.evaluate_string_pairs(prediction="Hello", prediction_b="Hi")
    >>> print(result)
    {'score': 0.5}
    """

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys of the chain.

        Returns:
            List[str]: The input keys.
        """
        return ["prediction", "prediction_b"]

    @property
    def evaluation_name(self) -> str:
        return f"pairwise_embedding_{self.distance_metric.value}_distance"

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Compute the score for two predictions.

        Args:
            inputs (Dict[str, Any]): The input data.
            run_manager (CallbackManagerForChainRun, optional):
                The callback manager.

        Returns:
            Dict[str, Any]: The computed score.
        """
        vectors = np.array(
            self.embeddings.embed_documents(
                [inputs["prediction"], inputs["prediction_b"]]
            )
        )
        score = self._compute_score(vectors)
        return {"score": score}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Asynchronously compute the score for two predictions.

        Args:
            inputs (Dict[str, Any]): The input data.
            run_manager (AsyncCallbackManagerForChainRun, optional):
                The callback manager.

        Returns:
            Dict[str, Any]: The computed score.
        """
        embedded = await self.embeddings.aembed_documents(
            [inputs["prediction"], inputs["prediction_b"]]
        )
        vectors = np.array(embedded)
        score = self._compute_score(vectors)
        return {"score": score}

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
        """Evaluate the embedding distance between two predictions.

        Args:
            prediction (str): The output string from the first model.
            prediction_b (str): The output string from the second model.
            callbacks (Callbacks, optional): The callbacks to use.
            tags (List[str], optional): Tags to apply to traces
            metadata (Dict[str, Any], optional): metadata to apply to
            **kwargs (Any): Additional keyword arguments.

        Returns:
            dict: A dictionary containing:
                - score: The embedding distance between the two
                    predictions.
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
        """Asynchronously evaluate the embedding distance

        between two predictions.

        Args:
            prediction (str): The output string from the first model.
            prediction_b (str): The output string from the second model.
            callbacks (Callbacks, optional): The callbacks to use.
            tags (List[str], optional): Tags to apply to traces
            metadata (Dict[str, Any], optional): metadata to apply to traces
            **kwargs (Any): Additional keyword arguments.

        Returns:
            dict: A dictionary containing:
                - score: The embedding distance between the two
                    predictions.
        """
        result = await self.acall(
            inputs={"prediction": prediction, "prediction_b": prediction_b},
            callbacks=callbacks,
            tags=tags,
            metadata=metadata,
            include_run_info=include_run_info,
        )
        return self._prepare_output(result)
