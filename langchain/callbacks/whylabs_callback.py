from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, Generation, LLMResult
from langchain.utils import get_from_env

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
    """WhyLabs CallbackHandler."""

    def __init__(self, logger: Logger):
        """Initiate the rolling logger"""
        super().__init__()
        self.logger = logger
        diagnostic_logger.info(
            "Initialized WhyLabs callback handler with configured whylogs Logger."
        )

    def _profile_generations(self, generations: List[Generation]) -> None:
        for gen in generations:
            self.logger.log({"response": gen.text})

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Pass the input prompts to the logger"""
        for prompt in prompts:
            self.logger.log({"prompt": prompt})

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Pass the generated response to the logger."""
        for generations in response.generations:
            self._profile_generations(generations)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Do nothing."""
        pass

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Do nothing."""

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Do nothing."""

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        pass

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Do nothing."""

    def on_agent_action(
        self, action: AgentAction, color: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Do nothing."""

    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Do nothing."""

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        pass

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Do nothing."""

    def on_agent_finish(
        self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Run on agent end."""
        pass

    def flush(self) -> None:
        self.logger._do_rollover()
        diagnostic_logger.info("Flushing WhyLabs logger, writing profile...")

    def close(self) -> None:
        self.logger.close()
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
    ) -> Logger:
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
        """
        # langkit library will import necessary whylogs libraries
        import_langkit(sentiment=sentiment, toxicity=toxicity, themes=themes)

        import whylogs as why
        from whylogs.api.writer.whylabs import WhyLabsWriter
        from whylogs.core.schema import DeclarativeSchema
        from whylogs.experimental.core.metrics.udf_metric import generate_udf_schema

        api_key = api_key or get_from_env("api_key", "WHYLABS_API_KEY")
        org_id = org_id or get_from_env("org_id", "WHYLABS_DEFAULT_ORG_ID")
        dataset_id = dataset_id or get_from_env(
            "dataset_id", "WHYLABS_DEFAULT_DATASET_ID"
        )
        whylabs_writer = WhyLabsWriter(
            api_key=api_key, org_id=org_id, dataset_id=dataset_id
        )

        langkit_schema = DeclarativeSchema(generate_udf_schema())
        whylabs_logger = why.logger(
            mode="rolling", interval=5, when="M", schema=langkit_schema
        )

        whylabs_logger.append_writer(writer=whylabs_writer)
        diagnostic_logger.info(
            "Started whylogs Logger with WhyLabsWriter and initialized LangKit. 📝"
        )
        return cls(whylabs_logger)
