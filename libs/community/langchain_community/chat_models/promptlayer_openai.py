"""PromptLayer wrapper."""

import datetime
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult

from langchain_community.chat_models import ChatOpenAI


class PromptLayerChatOpenAI(ChatOpenAI):
    """`PromptLayer` and `OpenAI` Chat large language models API.

    To use, you should have the ``openai`` and ``promptlayer`` python
    package installed, and the environment variable ``OPENAI_API_KEY``
    and ``PROMPTLAYER_API_KEY`` set with your openAI API key and
    promptlayer key respectively.

    All parameters that can be passed to the OpenAI LLM can also
    be passed here. The PromptLayerChatOpenAI adds to optional

    parameters:
        ``pl_tags``: List of strings to tag the request with.
        ``return_pl_id``: If True, the PromptLayer request ID will be
            returned in the ``generation_info`` field of the
            ``Generation`` object.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import PromptLayerChatOpenAI
            openai = PromptLayerChatOpenAI(model="gpt-3.5-turbo")
    """

    pl_tags: Optional[List[str]]
    return_pl_id: Optional[bool] = False

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Call ChatOpenAI generate and then call PromptLayer API to log the request."""
        from promptlayer.utils import get_api_key, promptlayer_api_request

        request_start_time = datetime.datetime.now().timestamp()
        generated_responses = super()._generate(
            messages, stop, run_manager, stream=stream, **kwargs
        )
        request_end_time = datetime.datetime.now().timestamp()
        message_dicts, params = super()._create_message_dicts(messages, stop)
        for i, generation in enumerate(generated_responses.generations):
            response_dict, params = super()._create_message_dicts(
                [generation.message], stop
            )
            params = {**params, **kwargs}
            pl_request_id = promptlayer_api_request(
                "langchain.PromptLayerChatOpenAI",
                "langchain",
                message_dicts,
                params,
                self.pl_tags,
                response_dict,
                request_start_time,
                request_end_time,
                get_api_key(),
                return_pl_id=self.return_pl_id,
            )
            if self.return_pl_id:
                if generation.generation_info is None or not isinstance(
                    generation.generation_info, dict
                ):
                    generation.generation_info = {}
                generation.generation_info["pl_request_id"] = pl_request_id
        return generated_responses

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Call ChatOpenAI agenerate and then call PromptLayer to log."""
        from promptlayer.utils import get_api_key, promptlayer_api_request_async

        request_start_time = datetime.datetime.now().timestamp()
        generated_responses = await super()._agenerate(
            messages, stop, run_manager, stream=stream, **kwargs
        )
        request_end_time = datetime.datetime.now().timestamp()
        message_dicts, params = super()._create_message_dicts(messages, stop)
        for i, generation in enumerate(generated_responses.generations):
            response_dict, params = super()._create_message_dicts(
                [generation.message], stop
            )
            params = {**params, **kwargs}
            pl_request_id = await promptlayer_api_request_async(
                "langchain.PromptLayerChatOpenAI.async",
                "langchain",
                message_dicts,
                params,
                self.pl_tags,
                response_dict,
                request_start_time,
                request_end_time,
                get_api_key(),
                return_pl_id=self.return_pl_id,
            )
            if self.return_pl_id:
                if generation.generation_info is None or not isinstance(
                    generation.generation_info, dict
                ):
                    generation.generation_info = {}
                generation.generation_info["pl_request_id"] = pl_request_id
        return generated_responses

    @property
    def _llm_type(self) -> str:
        return "promptlayer-openai-chat"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            **super()._identifying_params,
            "pl_tags": self.pl_tags,
            "return_pl_id": self.return_pl_id,
        }
