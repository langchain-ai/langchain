"""PromptLayer wrapper."""
import datetime
from typing import List, Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.llms import OpenAI, OpenAIChat
from langchain.schema import LLMResult


class PromptLayerOpenAI(OpenAI):
    """Wrapper around OpenAI large language models.

    To use, you should have the ``openai`` and ``promptlayer`` python
    package installed, and the environment variable ``OPENAI_API_KEY``
    and ``PROMPTLAYER_API_KEY`` set with your openAI API key and
    promptlayer key respectively.

    All parameters that can be passed to the OpenAI LLM can also
    be passed here. The PromptLayerOpenAI LLM adds two optional
    parameters:
        ``pl_tags``: List of strings to tag the request with.
        ``return_pl_id``: If True, the PromptLayer request ID will be
            returned in the ``generation_info`` field of the
            ``Generation`` object.

    Example:
        .. code-block:: python

            from langchain.llms import PromptLayerOpenAI
            openai = PromptLayerOpenAI(model_name="text-davinci-003")
    """

    pl_tags: Optional[List[str]]
    return_pl_id: Optional[bool] = False

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> LLMResult:
        """Call OpenAI generate and then call PromptLayer API to log the request."""
        from promptlayer.utils import get_api_key, promptlayer_api_request

        request_start_time = datetime.datetime.now().timestamp()
        generated_responses = super()._generate(prompts, stop, run_manager)
        request_end_time = datetime.datetime.now().timestamp()
        for i in range(len(prompts)):
            prompt = prompts[i]
            generation = generated_responses.generations[i][0]
            resp = {
                "text": generation.text,
                "llm_output": generated_responses.llm_output,
            }
            pl_request_id = promptlayer_api_request(
                "langchain.PromptLayerOpenAI",
                "langchain",
                [prompt],
                self._identifying_params,
                self.pl_tags,
                resp,
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
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    ) -> LLMResult:
        from promptlayer.utils import get_api_key, promptlayer_api_request_async

        request_start_time = datetime.datetime.now().timestamp()
        generated_responses = await super()._agenerate(prompts, stop, run_manager)
        request_end_time = datetime.datetime.now().timestamp()
        for i in range(len(prompts)):
            prompt = prompts[i]
            generation = generated_responses.generations[i][0]
            resp = {
                "text": generation.text,
                "llm_output": generated_responses.llm_output,
            }
            pl_request_id = await promptlayer_api_request_async(
                "langchain.PromptLayerOpenAI.async",
                "langchain",
                [prompt],
                self._identifying_params,
                self.pl_tags,
                resp,
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


class PromptLayerOpenAIChat(OpenAIChat):
    """Wrapper around OpenAI large language models.

    To use, you should have the ``openai`` and ``promptlayer`` python
    package installed, and the environment variable ``OPENAI_API_KEY``
    and ``PROMPTLAYER_API_KEY`` set with your openAI API key and
    promptlayer key respectively.

    All parameters that can be passed to the OpenAIChat LLM can also
    be passed here. The PromptLayerOpenAIChat adds two optional
    parameters:
        ``pl_tags``: List of strings to tag the request with.
        ``return_pl_id``: If True, the PromptLayer request ID will be
            returned in the ``generation_info`` field of the
            ``Generation`` object.

    Example:
        .. code-block:: python

            from langchain.llms import PromptLayerOpenAIChat
            openaichat = PromptLayerOpenAIChat(model_name="gpt-3.5-turbo")
    """

    pl_tags: Optional[List[str]]
    return_pl_id: Optional[bool] = False

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> LLMResult:
        """Call OpenAI generate and then call PromptLayer API to log the request."""
        from promptlayer.utils import get_api_key, promptlayer_api_request

        request_start_time = datetime.datetime.now().timestamp()
        generated_responses = super()._generate(prompts, stop, run_manager)
        request_end_time = datetime.datetime.now().timestamp()
        for i in range(len(prompts)):
            prompt = prompts[i]
            generation = generated_responses.generations[i][0]
            resp = {
                "text": generation.text,
                "llm_output": generated_responses.llm_output,
            }
            pl_request_id = promptlayer_api_request(
                "langchain.PromptLayerOpenAIChat",
                "langchain",
                [prompt],
                self._identifying_params,
                self.pl_tags,
                resp,
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
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    ) -> LLMResult:
        from promptlayer.utils import get_api_key, promptlayer_api_request_async

        request_start_time = datetime.datetime.now().timestamp()
        generated_responses = await super()._agenerate(prompts, stop, run_manager)
        request_end_time = datetime.datetime.now().timestamp()
        for i in range(len(prompts)):
            prompt = prompts[i]
            generation = generated_responses.generations[i][0]
            resp = {
                "text": generation.text,
                "llm_output": generated_responses.llm_output,
            }
            pl_request_id = await promptlayer_api_request_async(
                "langchain.PromptLayerOpenAIChat.async",
                "langchain",
                [prompt],
                self._identifying_params,
                self.pl_tags,
                resp,
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
