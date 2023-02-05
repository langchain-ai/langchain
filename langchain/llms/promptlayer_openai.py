import datetime
from langchain.llms import OpenAI
from langchain.schema import LLMResult
from promptlayer.utils import get_api_key, promptlayer_api_request
from pydantic import BaseModel
from typing import List, Optional

class PromptLayerOpenAI(OpenAI, BaseModel):
    pl_tags: Optional[List[str]]
    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> LLMResult: 
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