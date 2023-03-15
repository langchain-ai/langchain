"""PromptLayer wrapper."""
import datetime
from typing import List, Optional

from pydantic import BaseModel

from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseMessage, ChatResult


class PromptLayerChatOpenAI(ChatOpenAI, BaseModel):
    """Wrapper around OpenAI Chat large language models and PromptLayer.

    To use, you should have the ``openai`` and ``promptlayer`` python
    package installed, and the environment variable ``OPENAI_API_KEY``
    and ``PROMPTLAYER_API_KEY`` set with your openAI API key and
    promptlayer key respectively.

    All parameters that can be passed to the OpenAI LLM can also
    be passed here. The PromptLayerChatOpenAI LLM adds an extra
    ``pl_tags`` parameter that can be used to tag the request.

    Example:
        .. code-block:: python

            from langchain.chat_models import PromptLayerChatOpenAI
            openai = PromptLayerChatOpenAI(model_name="gpt-3.5-turbo")
    """

    pl_tags: Optional[List[str]]

    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> ChatResult:
        """Call ChatOpenAI generate and then call PromptLayer API to log the request."""
        from promptlayer.utils import get_api_key, promptlayer_api_request

        request_start_time = datetime.datetime.now().timestamp()
        generated_responses = super()._generate(messages, stop)
        request_end_time = datetime.datetime.now().timestamp()
        message_dicts, params = super()._create_message_dicts(messages, stop)
        for i, generation in enumerate(generated_responses.generations):
            response_dict, params = super()._create_message_dicts(
                [generation.message], stop
            )
            promptlayer_api_request(
                "langchain.PromptLayerChatOpenAI",
                "langchain",
                message_dicts,
                params,
                self.pl_tags,
                response_dict,
                request_start_time,
                request_end_time,
                get_api_key(),
            )
        return generated_responses

    async def _agenerate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> ChatResult:
        """Call ChatOpenAI agenerate and then call PromptLayer to log."""
        from promptlayer.utils import get_api_key, promptlayer_api_request

        request_start_time = datetime.datetime.now().timestamp()
        generated_responses = await super()._agenerate(messages, stop)
        request_end_time = datetime.datetime.now().timestamp()
        message_dicts, params = super()._create_message_dicts(messages, stop)
        for i, generation in enumerate(generated_responses.generations):
            response_dict, params = super()._create_message_dicts(
                [generation.message], stop
            )
            promptlayer_api_request(
                "langchain.PromptLayerChatOpenAI.async",
                "langchain",
                message_dicts,
                params,
                self.pl_tags,
                response_dict,
                request_start_time,
                request_end_time,
                get_api_key(),
            )
        return generated_responses
