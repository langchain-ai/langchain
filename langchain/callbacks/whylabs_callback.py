import logging
from typing import Any, Dict, List, Optional, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, Generation, LLMResult

logger = logging.getLogger(__name__)


def import_langkit(
    sentiment: Optional[bool] = None,
    toxicity: Optional[bool] = None,
    themes: Optional[bool] = None,
) -> Any:
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

    def __init__(
        self,
        api_key: Optional[str] = None,
        org_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        sentiment: Optional[bool] = None,
        toxicity: Optional[bool] = None,
        themes: Optional[bool] = None,
    ):
        """Initiate the rolling logger"""
        super().__init__()

        # langkit library will import necessary whylogs libraries
        import_langkit(sentiment=sentiment, toxicity=toxicity, themes=themes)

        import whylogs as why
        from whylogs.api.writer.whylabs import WhyLabsWriter
        from whylogs.core.schema import DeclarativeSchema
        from whylogs.experimental.core.metrics.udf_metric import generate_udf_schema

        self.writer = WhyLabsWriter(
            api_key=api_key, org_id=org_id, dataset_id=dataset_id
        )

        langkit_schema = DeclarativeSchema(generate_udf_schema())
        self.logger = why.logger(
            mode="rolling", interval=5, when="M", schema=langkit_schema
        )

        self.logger.append_writer(writer=self.writer)
        logger.info("Started WhyLabs callback handler and initialized LangKit. 📝")

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
        self.logger.close()
        logger.info("Closing WhyLabs logger, see you next time!")
