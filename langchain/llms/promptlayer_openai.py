"""PromptLayer wrapper."""
import datetime
from typing import List, Optional

from pydantic import BaseModel

from langchain.llms import OpenAI
from langchain.schema import LLMResult


class PromptLayerOpenAI(OpenAI, BaseModel):
    """Wrapper around OpenAI large language models.

    To use, you should have the ``openai`` and ``promptlayer`` python
    package installed, and the environment variable ``OPENAI_API_KEY``
    and ``PROMPTLAYER_API_KEY`` set with your openAI API key and
    promptlayer key respectively.

    All parameters that can be passed to the OpenAI LLM can also
    be passed here. The PromptLayerOpenAI LLM adds an extra
    ``pl_tags`` parameter that can be used to tag the request.

    Example:
        .. code-block:: python

            from langchain.llms import OpenAI
            openai = OpenAI(model_name="text-davinci-003")
    """

    pl_tags: Optional[List[str]]

    def _generate(
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> LLMResult:
        """Call OpenAI generate and then call PromptLayer API to log the request."""
        from promptlayer.utils import get_api_key, promptlayer_api_request

        request_start_time = datetime.datetime.now().timestamp()
        generated_responses = super()._generate(prompts, stop)
        request_end_time = datetime.datetime.now().timestamp()
        for i in range(len(prompts)):
            prompt = prompts[i]
            resp = generated_responses.generations[i]
            promptlayer_api_request(
                "langchain.PromptLayerOpenAI",
                "langchain",
                [prompt],
                self._identifying_params,
                self.pl_tags,
                resp[0].text,
                request_start_time,
                request_end_time,
                get_api_key(),
            )
        return generated_responses

    async def _agenerate(
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> LLMResult:
        from promptlayer.utils import get_api_key, promptlayer_api_request

        request_start_time = datetime.datetime.now().timestamp()
        generated_responses = await super()._agenerate(prompts, stop)
        request_end_time = datetime.datetime.now().timestamp()
        for i in range(len(prompts)):
            prompt = prompts[i]
            resp = generated_responses.generations[i]
            promptlayer_api_request(
                "langchain.PromptLayerOpenAI.async",
                "langchain",
                [prompt],
                self._identifying_params,
                self.pl_tags,
                resp[0].text,
                request_start_time,
                request_end_time,
                get_api_key(),
            )
        return generated_responses
