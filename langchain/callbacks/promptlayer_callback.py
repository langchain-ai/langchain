import os
import datetime
from typing import Dict, Any, List

import promptlayer
from promptlayer.utils import get_api_key, promptlayer_api_request

from langchain.llms import OpenAI, Replicate
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, LLMResult
from langchain.callbacks.base import BaseCallbackHandler


class PromptLayerHandler(BaseCallbackHandler):
    # TODO: Do I need this function?
    # def on_chat_model_start(serialized, messages, run_id, parent_run_id)

    def __init__(self, request_id_func=None, pl_tags=[]):
        self.request_id_func = request_id_func
        self.pl_tags = pl_tags

        self.request_start_time = None
        self.request_end_time = None

        self.prompts = None
        self.name = None
        self.invocation_params = None


    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        self.request_start_time = datetime.datetime.now().timestamp()
        self.prompts = prompts
        self.invocation_params = kwargs.get('invocation_params', {})
        self.name = self.invocation_params.get('_type', 'No Type')


    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        self.request_end_time = datetime.datetime.now().timestamp()

        for i in range(len(response.generations)):
            generation = response.generations[i][0]

            resp = {
                "text": generation.text,
                "llm_output": response.llm_output,
            }
            pl_request_id = promptlayer_api_request(
                f"langchain.{self.name}",
                "langchain",
                [self.prompts[i]],
                self.invocation_params,
                self.pl_tags,
                resp,
                self.request_start_time,
                self.request_end_time,
                get_api_key(),
                return_pl_id=bool(self.request_id_func),
            )

            if self.request_id_func:
                self.request_id_func(pl_request_id)


# TODO: Move!!!
# Test Section
l = []
def request_id_func(request_id):
    print(l)
    l.append(request_id)


# Test OpenAI
openai_llm = OpenAI(
    model_name="text-davinci-002",
    callbacks=[PromptLayerHandler(
        request_id_func = request_id_func,
        pl_tags = ["OPENAI WORKS!"]
    )]
)
llm_results = openai_llm.generate([
    "Tell me a joke 1",
    "Where is Ohio? 2",
    "Where is Ohio? 3",
])

# Test Replicate LLM
replicate_llm = Replicate(
    model="replicate/dolly-v2-12b:ef0e1aefc61f8e096ebe4db6b2bacc297daf2ef6899f0f7e001ec445893500e5",
    callbacks=[PromptLayerHandler(
        request_id_func = request_id_func,
        pl_tags = ["REPLICATE WORKS!"]
    )]
)
llm_results = replicate_llm.generate([
    "Tell me a joke 1",
    "Where is Ohio? 2",
    "Where is Ohio? 3",
])

# Test OpenAIChat LLM
chat_llm = ChatOpenAI(
    temperature=0,
    callbacks=[PromptLayerHandler(
        request_id_func = request_id_func,
        pl_tags = ["OpenAIChat WORKS!"]
    )]
)
llm_results = chat_llm([
    HumanMessage(content="What comes after 1,2,3 ?")
])
