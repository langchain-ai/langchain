import os
import warnings
from typing import Any, Dict, List, Optional, Union

from packaging.version import parse

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult


class DeepevalCallbackHandler(BaseCallbackHandler):
    """Callback Handler that logs into deepeval.

    Args:
        implementation_name: name of the `FeedbackDataset` in deepeval. Note that it must
            exist in advance. If you need help on how to create a `FeedbackDataset` in
            deepeval, please visit
            https://docs.deepeval.io/en/latest/guides/llms/practical_guides/use_deepeval_callback_in_langchain.html.
        api_key: API Key to connect to the deepeval server. Defaults to `None`, which
            means that either `deepeval_API_KEY` environment variable or the default
            will be used.
        metrics: A list of metrics

    Raises:
        ImportError: if the `deepeval` package is not installed.
        ConnectionError: if the connection to deepeval fails.
        FileNotFoundError: if the `FeedbackDataset` retrieval from deepeval fails.

    Examples:
        >>> from langchain.llms import OpenAI
        >>> from langchain.callbacks import deepevalCallbackHandler
        >>> deepeval_callback = deepevalCallbackHandler(
        ...     implementation_name="exampleImplementation",
        ...     api_key="<apikey>",
        ... )
        >>> llm = OpenAI(
        ...     temperature=0,
        ...     callbacks=[deepeval_callback],
        ...     verbose=True,
        ...     openai_api_key="API_KEY_HERE",
        ... )
        >>> llm.generate([
        ...     "What is the best evaluation tool out there? (no bias at all)",
        ... ])
        "Deepeval, no doubt about it."
    """

    REPO_URL: str = "https://github.com/confident-ai/deepeval"
    ISSUES_URL: str = f"{REPO_URL}/issues"
    BLOG_URL: str = "https://docs.confident-ai.com"  # noqa: E501

    def __init__(
        self,
        api_key: Optional[str] = None,
        implementation_name: Optional[str] = None,
        metrics: List[Any] = None,
    ) -> None:
        """Initializes the `deepevalCallbackHandler`.

        Args:
            dataset_name: name of the `FeedbackDataset` in deepeval. Note that it must
                exist in advance. If you need help on how to create a `FeedbackDataset`
                in deepeval, please visit
                https://docs.deepeval.io/en/latest/guides/llms/practical_guides/use_deepeval_callback_in_langchain.html.
            workspace_name: name of the workspace in deepeval where the specified
                `FeedbackDataset` lives in. Defaults to `None`, which means that the
                default workspace will be used.
            api_url: URL of the deepeval Server that we want to use, and where the
                `FeedbackDataset` lives in. Defaults to `None`, which means that either
                `deepeval_API_URL` environment variable or the default will be used.
            api_key: API Key to connect to the deepeval Server. Defaults to `None`, which
                means that either `deepeval_API_KEY` environment variable or the default
                will be used.

        Raises:
            ImportError: if the `deepeval` package is not installed.
            ConnectionError: if the connection to deepeval fails.
            FileNotFoundError: if the `FeedbackDataset` retrieval from deepeval fails.
        """

        super().__init__()

        # Import deepeval (not via `import_deepeval` to keep hints in IDEs)
        try:
            import deepeval
        except ImportError:
            raise ImportError(
                "To use the deepeval callback manager you need to have the `deepeval` "
                "Python package installed. Please install it with `pip install deepeval`"
            )

        if os.path.exists(".deepeval"):
            warnings.warn(
                """You are currently not logging anything to the dashboard, we recommend using `deepeval login`."""
            )

        # Set the deepeval variables
        self.implementation_name = implementation_name
        self.metrics = metrics
        self.api_key = api_key

        warnings.warn(
            (
                "The `DeepEvalCallbackHandler` is currently in beta and is subject to"
                " change based on updates to `langchain`. Please report any issues to"
                f" {self.ISSUES_URL} as an `integration` issue."
            ),
        )

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Store the prompts"""
        self.prompts = prompts

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Do nothing when a new token is generated."""
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Log records to deepeval when an LLM ends."""
        from deepeval.metrics.metric import Metric
        from deepeval.metrics.answer_relevancy import AnswerRelevancy
        from deepeval.metrics.bias_classifier import UnBiasedMetric
        from deepeval.metrics.toxic_classifier import NonToxicMetric

        for metric in self.metrics:
            for i, generation in enumerate(response.generations):
                metric: Metric
                # Here, we only measure the first generation's output
                output = generation[0].text
                query = self.prompts[i]
                if isinstance(metric, AnswerRelevancy):
                    result = metric.measure(
                        output=output,
                        query=query,
                    )
                    print(f"Answer Relevancy: {result}")
                elif isinstance(metric, UnBiasedMetric):
                    metric = UnBiasedMetric(model_name="original", minimum_score=0.5)
                    score = metric.measure(output)
                    print(f"Bias Score: {score}")
                elif isinstance(metric, NonToxicMetric):
                    metric = NonToxicMetric(minimum_score=0.5)
                    score = metric.measure(output)
                    print(f"Toxic Score: {score}")
                else:
                    raise ValueError(
                        f"Metric {metric.__name__} is not supported by deepeval callbacks."
                    )

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing when LLM outputs an error."""
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Do nothing when chain starts"""
        pass

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Do nothing when chain ends."""
        pass

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing when LLM chain outputs an error."""
        pass

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Do nothing when tool starts."""
        pass

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Do nothing when agent takes a specific action."""
        pass

    def on_tool_end(
        self,
        output: str,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Do nothing when tool ends."""
        pass

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing when tool outputs an error."""
        pass

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Do nothing"""
        pass

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Do nothing"""
        pass
