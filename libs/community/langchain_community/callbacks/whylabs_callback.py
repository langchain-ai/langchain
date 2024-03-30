from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.utils import get_from_env

if TYPE_CHECKING:
    from whylogs.api.logger.logger import Logger

diagnostic_logger = logging.getLogger(__name__)


def import_langkit(
    sentiment: bool = False,
    toxicity: bool = False,
    themes: bool = False,
) -> Any:
    """Import the langkit python package and raise an error if it is not installed.

    Args:
        sentiment: Whether to import the langkit.sentiment module. Defaults to False.
        toxicity: Whether to import the langkit.toxicity module. Defaults to False.
        themes: Whether to import the langkit.themes module. Defaults to False.

    Returns:
        The imported langkit module.
    """
    try:
        import langkit  # noqa: F401
        import langkit.regexes  # noqa: F401
        import langkit.textstat  # noqa: F401

        if sentiment:
            import langkit.sentiment  # noqa: F401
        if toxicity:
            import langkit.toxicity  # noqa: F401
        if themes:
            import langkit.themes  # noqa: F401
    except ImportError:
        raise ImportError(
            "To use the whylabs callback manager you need to have the `langkit` python "
            "package installed. Please install it with `pip install langkit`."
        )
    return langkit


class WhyLabsCallbackHandler(BaseCallbackHandler):
    """
    Callback Handler for logging to WhyLabs. This callback handler utilizes
    `langkit` to extract features from the prompts & responses when interacting with
    an LLM. These features can be used to guardrail, evaluate, and observe interactions
    over time to detect issues relating to hallucinations, prompt engineering,
    or output validation. LangKit is an LLM monitoring toolkit developed by WhyLabs.

    Here are some examples of what can be monitored with LangKit:
    * Text Quality
      - readability score
      - complexity and grade scores
    * Text Relevance
      - Similarity scores between prompt/responses
      - Similarity scores against user-defined themes
      - Topic classification
    * Security and Privacy
      - patterns - count of strings matching a user-defined regex pattern group
      - jailbreaks - similarity scores with respect to known jailbreak attempts
      - prompt injection - similarity scores with respect to known prompt attacks
      - refusals - similarity scores with respect to known LLM refusal responses
    * Sentiment and Toxicity
      - sentiment analysis
      - toxicity analysis

    For more information, see https://docs.whylabs.ai/docs/language-model-monitoring
    or check out the LangKit repo here: https://github.com/whylabs/langkit

    ---
    Args:
        api_key (Optional[str]): WhyLabs API key. Optional because the preferred
            way to specify the API key is with environment variable
            WHYLABS_API_KEY.
        org_id (Optional[str]): WhyLabs organization id to write profiles to.
            Optional because the preferred way to specify the organization id is
            with environment variable WHYLABS_DEFAULT_ORG_ID.
        dataset_id (Optional[str]): WhyLabs dataset id to write profiles to.
            Optional because the preferred way to specify the dataset id is
            with environment variable WHYLABS_DEFAULT_DATASET_ID.
        sentiment (bool): Whether to enable sentiment analysis. Defaults to False.
        toxicity (bool): Whether to enable toxicity analysis. Defaults to False.
        themes (bool): Whether to enable theme analysis. Defaults to False.
    """

    def __init__(self, logger: Logger, handler: Any):
        """Initiate the rolling logger."""
        super().__init__()
        if hasattr(handler, "init"):
            handler.init(self)
        if hasattr(handler, "_get_callbacks"):
            self._callbacks = handler._get_callbacks()
        else:
            self._callbacks = dict()
            diagnostic_logger.warning("initialized handler without callbacks.")
        self._logger = logger

    def flush(self) -> None:
        """Explicitly write current profile if using a rolling logger."""
        if self._logger and hasattr(self._logger, "_do_rollover"):
            self._logger._do_rollover()
            diagnostic_logger.info("Flushing WhyLabs logger, writing profile...")

    def close(self) -> None:
        """Close any loggers to allow writing out of any profiles before exiting."""
        if self._logger and hasattr(self._logger, "close"):
            self._logger.close()
            diagnostic_logger.info("Closing WhyLabs logger, see you next time!")

    def __enter__(self) -> WhyLabsCallbackHandler:
        return self

    def __exit__(
        self, exception_type: Any, exception_value: Any, traceback: Any
    ) -> None:
        self.close()

    @classmethod
    def from_params(
        cls,
        *,
        api_key: Optional[str] = None,
        org_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        sentiment: bool = False,
        toxicity: bool = False,
        themes: bool = False,
        logger: Optional[Logger] = None,
    ) -> WhyLabsCallbackHandler:
        """Instantiate whylogs Logger from params.

        Args:
            api_key (Optional[str]): WhyLabs API key. Optional because the preferred
                way to specify the API key is with environment variable
                WHYLABS_API_KEY.
            org_id (Optional[str]): WhyLabs organization id to write profiles to.
                If not set must be specified in environment variable
                WHYLABS_DEFAULT_ORG_ID.
            dataset_id (Optional[str]): The model or dataset this callback is gathering
                telemetry for. If not set must be specified in environment variable
                WHYLABS_DEFAULT_DATASET_ID.
            sentiment (bool): If True will initialize a model to perform
                sentiment analysis compound score. Defaults to False and will not gather
                this metric.
            toxicity (bool): If True will initialize a model to score
                toxicity. Defaults to False and will not gather this metric.
            themes (bool): If True will initialize a model to calculate
                distance to configured themes. Defaults to None and will not gather this
                metric.
            logger (Optional[Logger]): If specified will bind the configured logger as
                the telemetry gathering agent. Defaults to LangKit schema with periodic
                WhyLabs writer.
        """
        # langkit library will import necessary whylogs libraries
        import_langkit(sentiment=sentiment, toxicity=toxicity, themes=themes)

        import whylogs as why
        from langkit.callback_handler import get_callback_instance
        from whylogs.api.writer.whylabs import WhyLabsWriter
        from whylogs.experimental.core.udf_schema import udf_schema

        if logger is None:
            api_key = api_key or get_from_env("api_key", "WHYLABS_API_KEY")
            org_id = org_id or get_from_env("org_id", "WHYLABS_DEFAULT_ORG_ID")
            dataset_id = dataset_id or get_from_env(
                "dataset_id", "WHYLABS_DEFAULT_DATASET_ID"
            )
            whylabs_writer = WhyLabsWriter(
                api_key=api_key, org_id=org_id, dataset_id=dataset_id
            )

            whylabs_logger = why.logger(
                mode="rolling", interval=5, when="M", schema=udf_schema()
            )

            whylabs_logger.append_writer(writer=whylabs_writer)
        else:
            diagnostic_logger.info("Using passed in whylogs logger {logger}")
            whylabs_logger = logger

        callback_handler_cls = get_callback_instance(logger=whylabs_logger, impl=cls)
        diagnostic_logger.info(
            "Started whylogs Logger with WhyLabsWriter and initialized LangKit. 📝"
        )
        return callback_handler_cls
