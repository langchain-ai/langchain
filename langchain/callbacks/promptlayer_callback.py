import os
import datetime
from typing import Dict, Any, List, Optional

from promptlayer.utils import get_api_key, promptlayer_api_request

from langchain.llms import OpenAI, Replicate
from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage, AIMessage, LLMResult, UUID, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler


class PromptLayerHandler(BaseCallbackHandler):
    def _default_params(self):
        # TODO: Should I be setting defaults here for PL?
        return {
            "model_name": "gpt-4",
            "temperature": 0,
        }


    def _convert_message_to_dict(self, message):
        if isinstance(message, HumanMessage):
            message_dict = {"role": "user", "content": message.content}
        elif isinstance(message, AIMessage):
            message_dict = {"role": "assistant", "content": message.content}
        elif isinstance(message, SystemMessage):
            message_dict = {"role": "system", "content": message.content}
        else:
            raise ValueError(f"Got unknown type {message}")
        if "name" in message.additional_kwargs:
            message_dict["name"] = message.additional_kwargs["name"]
        return message_dict


    def _create_message_dicts(self, messages):
        params: Dict[str, Any] = self._default_params()
        message_dicts = [self._convert_message_to_dict(m) for m in messages[0]]
        return message_dicts, params


    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        self.request_start_time = datetime.datetime.now().timestamp()

        self.messages = messages
        self.invocation_params = kwargs.get('invocation_params', {})
        self.name = self.invocation_params.get('_type', 'No Type')

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

            if self.name == 'openai-chat':
                function_name = f"openai.ChatCompletion.create"
                message_dicts, model_params = self._create_message_dicts(self.messages)
                model_input = []
                model_response, model_params = self._create_message_dicts([[generation.message]])
                model_response = {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "index": 0,
                            "message": {
                                "content": generation.message.content,
                                "role": "assistant"
                            }
                        }
                    ],
                    "usage": response.llm_output.get('token_usage', {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0})
                }
                model_params['messages'] = message_dicts
            else:
                function_name = f"langchain.OpenAI"
                model_input = [self.prompts[i]]
                model_params = self.invocation_params
                model_response = resp

            # import pdb; pdb.set_trace()
            pl_request_id = promptlayer_api_request(
                function_name, # TODO: should be self.name but would break promptlayerUI. Should add catchall
                "langchain",
                model_input,
                model_params,
                self.pl_tags,
                model_response,
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
    HumanMessage(content="What comes after 1,2,3 ?"),
    HumanMessage(content="Tell me another joke?"),
])
