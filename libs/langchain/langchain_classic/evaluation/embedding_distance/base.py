"""A chain for comparing the output of two models using embeddings."""

import functools
import logging
from enum import Enum
from importlib import util
from typing import Any

from langchain_core.callbacks import Callbacks
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.embeddings import Embeddings
from langchain_core.utils import pre_init
from pydantic import ConfigDict, Field
from typing_extensions import override

from langchain_classic.chains.base import Chain
from langchain_classic.evaluation.schema import PairwiseStringEvaluator, StringEvaluator
from langchain_classic.schema import RUN_KEY


def _import_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as e:
        msg = "Could not import numpy, please install with `pip install numpy`."
        raise ImportError(msg) from e
    return np


logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _check_numpy() -> bool:
    if bool(util.find_spec("numpy")):
        return True
    logger.warning(
        "NumPy not found in the current Python environment. "
        "langchain will use a pure Python implementation for embedding distance "
        "operations, which may significantly impact performance, especially for large "
        "datasets. For optimal speed and efficiency, consider installing NumPy: "
        "pip install numpy",
    )
    return False


def _embedding_factory() -> Embeddings:
    """Create an `Embeddings` object.

    Returns:
        The created `Embeddings` object.
    """
    # Here for backwards compatibility.
    # Generally, we do not want to be seeing imports from langchain community
    # or partner packages in langchain.
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        try:
            from langchain_community.embeddings.openai import (  # type: ignore[no-redef]
                OpenAIEmbeddings,
            )
        except ImportError as e:
            msg = (
                "Could not import OpenAIEmbeddings. Please install the "
                "OpenAIEmbeddings package using `pip install langchain-openai`."
            )
            raise ImportError(msg) from e
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
        embeddings: The embedding objects to vectorize the outputs.
        distance_metric: The distance metric to use for comparing the embeddings.
    """

    embeddings: Embeddings = Field(default_factory=_embedding_factory)
    distance_metric: EmbeddingDistance = Field(default=EmbeddingDistance.COSINE)

    @pre_init
    def _validate_tiktoken_installed(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Validate that the TikTok library is installed.

        Args:
            values: The values to validate.

        Returns:
            The validated values.
        """
        embeddings = values.get("embeddings")
        types_ = []
        try:
            from langchain_openai import OpenAIEmbeddings

            types_.append(OpenAIEmbeddings)
        except ImportError:
            pass

        try:
            from langchain_community.embeddings.openai import (  # type: ignore[no-redef]
                OpenAIEmbeddings,
            )

            types_.append(OpenAIEmbeddings)
        except ImportError:
            pass

        if not types_:
            msg = (
                "Could not import OpenAIEmbeddings. Please install the "
                "OpenAIEmbeddings package using `pip install langchain-openai`."
            )
            raise ImportError(msg)

        if isinstance(embeddings, tuple(types_)):
            try:
                import tiktoken  # noqa: F401
            except ImportError as e:
                msg = (
                    "The tiktoken library is required to use the default "
                    "OpenAI embeddings with embedding distance evaluators."
                    " Please either manually select a different Embeddings object"
                    " or install tiktoken using `pip install tiktoken`."
                )
                raise ImportError(msg) from e
        return values

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @property
    def output_keys(self) -> list[str]:
        """Return the output keys of the chain.

        Returns:
            The output keys.
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
            metric: The metric name.

        Returns:
            The metric function.
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
        msg = f"Invalid metric: {metric}"
        raise ValueError(msg)

    @staticmethod
    def _cosine_distance(a: Any, b: Any) -> Any:
        """Compute the cosine distance between two vectors.

        Args:
            a (np.ndarray): The first vector.
            b (np.ndarray): The second vector.

        Returns:
            np.ndarray: The cosine distance.
        """
        try:
            from langchain_core.vectorstores.utils import _cosine_similarity

            return 1.0 - _cosine_similarity(a, b)
        except ImportError:
            # Fallback to scipy if available
            try:
                from scipy.spatial.distance import cosine

                return cosine(a.flatten(), b.flatten())
            except ImportError:
                # Pure numpy fallback
                if _check_numpy():
                    np = _import_numpy()
                    a_flat = a.flatten()
                    b_flat = b.flatten()
                    dot_product = np.dot(a_flat, b_flat)
                    norm_a = np.linalg.norm(a_flat)
                    norm_b = np.linalg.norm(b_flat)
                    if norm_a == 0 or norm_b == 0:
                        return 0.0
                    return 1.0 - (dot_product / (norm_a * norm_b))
                # Pure Python implementation
                a_flat = a if hasattr(a, "__len__") else [a]
                b_flat = b if hasattr(b, "__len__") else [b]
                if hasattr(a, "flatten"):
                    a_flat = a.flatten()
                if hasattr(b, "flatten"):
                    b_flat = b.flatten()

                dot_product = sum(x * y for x, y in zip(a_flat, b_flat, strict=False))
                norm_a = sum(x * x for x in a_flat) ** 0.5
                norm_b = sum(x * x for x in b_flat) ** 0.5
                if norm_a == 0 or norm_b == 0:
                    return 0.0
                return 1.0 - (dot_product / (norm_a * norm_b))

    @staticmethod
    def _euclidean_distance(a: Any, b: Any) -> Any:
        """Compute the Euclidean distance between two vectors.

        Args:
            a (np.ndarray): The first vector.
            b (np.ndarray): The second vector.

        Returns:
            np.floating: The Euclidean distance.
        """
        try:
            from scipy.spatial.distance import euclidean

            return euclidean(a.flatten(), b.flatten())
        except ImportError:
            if _check_numpy():
                import numpy as np

                return np.linalg.norm(a - b)

            return sum((x - y) * (x - y) for x, y in zip(a, b, strict=False)) ** 0.5

    @staticmethod
    def _manhattan_distance(a: Any, b: Any) -> Any:
        """Compute the Manhattan distance between two vectors.

        Args:
            a (np.ndarray): The first vector.
            b (np.ndarray): The second vector.

        Returns:
            np.floating: The Manhattan distance.
        """
        try:
            from scipy.spatial.distance import cityblock

            return cityblock(a.flatten(), b.flatten())
        except ImportError:
            if _check_numpy():
                np = _import_numpy()
                return np.sum(np.abs(a - b))

            return sum(abs(x - y) for x, y in zip(a, b, strict=False))

    @staticmethod
    def _chebyshev_distance(a: Any, b: Any) -> Any:
        """Compute the Chebyshev distance between two vectors.

        Args:
            a (np.ndarray): The first vector.
            b (np.ndarray): The second vector.

        Returns:
            np.floating: The Chebyshev distance.
        """
        try:
            from scipy.spatial.distance import chebyshev

            return chebyshev(a.flatten(), b.flatten())
        except ImportError:
            if _check_numpy():
                np = _import_numpy()
                return np.max(np.abs(a - b))

            return max(abs(x - y) for x, y in zip(a, b, strict=False))

    @staticmethod
    def _hamming_distance(a: Any, b: Any) -> Any:
        """Compute the Hamming distance between two vectors.

        Args:
            a (np.ndarray): The first vector.
            b (np.ndarray): The second vector.

        Returns:
            np.floating: The Hamming distance.
        """
        try:
            from scipy.spatial.distance import hamming

            return hamming(a.flatten(), b.flatten())
        except ImportError:
            if _check_numpy():
                np = _import_numpy()
                return np.mean(a != b)

            return sum(1 for x, y in zip(a, b, strict=False) if x != y) / len(a)

    def _compute_score(self, vectors: Any) -> float:
        """Compute the score based on the distance metric.

        Args:
            vectors (np.ndarray): The input vectors.

        Returns:
            The computed score.
        """
        metric = self._get_metric(self.distance_metric)
        if _check_numpy() and isinstance(vectors, _import_numpy().ndarray):
            score = metric(vectors[0].reshape(1, -1), vectors[1].reshape(1, -1)).item()
        else:
            score = metric(vectors[0], vectors[1])
        return float(score)


class EmbeddingDistanceEvalChain(_EmbeddingDistanceChainMixin, StringEvaluator):
    """Embedding distance evaluation chain.

    Use embedding distances to score semantic difference between
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
            True if a reference is required, `False` otherwise.
        """
        return True

    @property
    @override
    def evaluation_name(self) -> str:
        return f"embedding_{self.distance_metric.value}_distance"

    @property
    def input_keys(self) -> list[str]:
        """Return the input keys of the chain.

        Returns:
            The input keys.
        """
        return ["prediction", "reference"]

    @override
    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        """Compute the score for a prediction and reference.

        Args:
            inputs: The input data.
            run_manager: The callback manager.

        Returns:
            The computed score.
        """
        vectors = self.embeddings.embed_documents(
            [inputs["prediction"], inputs["reference"]],
        )
        if _check_numpy():
            np = _import_numpy()
            vectors = np.array(vectors)
        score = self._compute_score(vectors)
        return {"score": score}

    @override
    async def _acall(
        self,
        inputs: dict[str, Any],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        """Asynchronously compute the score for a prediction and reference.

        Args:
            inputs: The input data.
            run_manager: The callback manager.

        Returns:
            The computed score.
        """
        vectors = await self.embeddings.aembed_documents(
            [
                inputs["prediction"],
                inputs["reference"],
            ],
        )
        if _check_numpy():
            np = _import_numpy()
            vectors = np.array(vectors)
        score = self._compute_score(vectors)
        return {"score": score}

    @override
    def _evaluate_strings(
        self,
        *,
        prediction: str,
        reference: str | None = None,
        callbacks: Callbacks = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        include_run_info: bool = False,
        **kwargs: Any,
    ) -> dict:
        """Evaluate the embedding distance between a prediction and reference.

        Args:
            prediction: The output string from the first model.
            reference: The output string from the second model.
            callbacks: The callbacks to use.
            tags: The tags to apply.
            metadata: The metadata to use.
            include_run_info: Whether to include run information in the output.
            **kwargs: Additional keyword arguments.

        Returns:
            `dict` containing:
                - score: The embedding distance between the two predictions.
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
        callbacks: Callbacks = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        include_run_info: bool = False,
        **kwargs: Any,
    ) -> dict:
        """Evaluate the embedding distance between a prediction and reference.

        Args:
            prediction: The output string from the first model.
            reference: The output string from the second model.
            callbacks: The callbacks to use.
            tags: The tags to apply.
            metadata: The metadata to use.
            include_run_info: Whether to include run information in the output.
            **kwargs: Additional keyword arguments.

        Returns:
            `dict` containing:
                - score: The embedding distance between the two predictions.
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
    _EmbeddingDistanceChainMixin,
    PairwiseStringEvaluator,
):
    """Use embedding distances to score semantic difference between two predictions.

    Examples:
    >>> chain = PairwiseEmbeddingDistanceEvalChain()
    >>> result = chain.evaluate_string_pairs(prediction="Hello", prediction_b="Hi")
    >>> print(result)
    {'score': 0.5}
    """

    @property
    def input_keys(self) -> list[str]:
        """Return the input keys of the chain.

        Returns:
            The input keys.
        """
        return ["prediction", "prediction_b"]

    @property
    def evaluation_name(self) -> str:
        """Return the evaluation name."""
        return f"pairwise_embedding_{self.distance_metric.value}_distance"

    @override
    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        """Compute the score for two predictions.

        Args:
            inputs: The input data.
            run_manager: The callback manager.

        Returns:
            The computed score.
        """
        vectors = self.embeddings.embed_documents(
            [
                inputs["prediction"],
                inputs["prediction_b"],
            ],
        )
        if _check_numpy():
            np = _import_numpy()
            vectors = np.array(vectors)
        score = self._compute_score(vectors)
        return {"score": score}

    @override
    async def _acall(
        self,
        inputs: dict[str, Any],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        """Asynchronously compute the score for two predictions.

        Args:
            inputs: The input data.
            run_manager: The callback manager.

        Returns:
            The computed score.
        """
        vectors = await self.embeddings.aembed_documents(
            [
                inputs["prediction"],
                inputs["prediction_b"],
            ],
        )
        if _check_numpy():
            np = _import_numpy()
            vectors = np.array(vectors)
        score = self._compute_score(vectors)
        return {"score": score}

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
        """Evaluate the embedding distance between two predictions.

        Args:
            prediction: The output string from the first model.
            prediction_b: The output string from the second model.
            callbacks: The callbacks to use.
            tags: The tags to apply.
            metadata: The metadata to use.
            include_run_info: Whether to include run information in the output.
            **kwargs: Additional keyword arguments.

        Returns:
            `dict` containing:
                - score: The embedding distance between the two predictions.
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
        """Asynchronously evaluate the embedding distance between two predictions.

        Args:
            prediction: The output string from the first model.
            prediction_b: The output string from the second model.
            callbacks: The callbacks to use.
            tags: The tags to apply.
            metadata: The metadata to use.
            include_run_info: Whether to include run information in the output.
            **kwargs: Additional keyword arguments.

        Returns:
            `dict` containing:
                - score: The embedding distance between the two predictions.
        """
        result = await self.acall(
            inputs={"prediction": prediction, "prediction_b": prediction_b},
            callbacks=callbacks,
            tags=tags,
            metadata=metadata,
            include_run_info=include_run_info,
        )
        return self._prepare_output(result)
