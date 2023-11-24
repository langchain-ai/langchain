## NOTE: This class is intentionally implemented to subclass either ChatModel or LLM for
##  demonstrative purposes and to make it function as a simple standalone file.

from langchain.callbacks.manager import CallbackManager, AsyncCallbackManager, AsyncCallbackManagerForLLMRun
from langchain.schema.output import GenerationChunk, ChatGenerationChunk
from langchain.schema.messages import BaseMessage, ChatMessageChunk
from langchain.pydantic_v1 import BaseModel, Field, root_validator
from langchain.utils import get_from_dict_or_env

try:    ## if running as part of package
    from .base import LLM
    STANDALONE = False
except: ## if running as standalone file
    from langchain.llms.base import LLM
    from langchain.chat_models.base import SimpleChatModel
    STANDALONE = True

import requests
import aiohttp
import asyncio
import json
import re
import os

from typing import Callable, Any, Dict, List, Optional, Tuple, Sequence, Union, Generator, AsyncIterator, Iterator
from json.decoder import JSONDecodeError
from requests.models import Response
from functools import partial

import logging
logger = logging.getLogger(__name__)

## TODO: Pull these in from ngc if possible
## Staging Options
AI_PLAY_URLS_STG = {
    'llama-13B-code'    : "https://stg.api.nvcf.nvidia.com/v2/nvcf/pexec/functions/eb1100de-60bf-4e9a-8617-b7d4652e0c37",
    'llama-34B-code'    : "https://stg.api.nvcf.nvidia.com/v2/nvcf/pexec/functions/1b739bf1-f3b0-4f04-9ace-3a1cbcefca49",
    'llama-2-13B-chat'  : "https://stg.api.nvcf.nvidia.com/v2/nvcf/pexec/functions/4e38f372-2533-4e76-b3d2-8bfe7b9ca783",
    'llama-2-70B-chat'  : "https://stg.api.nvcf.nvidia.com/v2/nvcf/pexec/functions/ee2643e0-bda6-4ddc-9841-378a11f1dd04",
    'mistral-7B-inst'   : "https://stg.api.nvcf.nvidia.com/v2/nvcf/pexec/functions/33809adf-b2d7-497b-aa21-83e3f334a953",
    # 'neva-22b'          : "https://stg.api.nvcf.nvidia.com/v2/nvcf/pexec/functions/7c94dd3e-72d7-48de-b0fe-aaceb5aa0c23",
    # 'fuyu-8b'           : "https://stg.api.nvcf.nvidia.com/v2/nvcf/pexec/functions/8f0f7c3e-7e31-40f1-abe6-d3d7c526fb7b"
}

## Production Options
AI_PLAY_URLS = {
    'llama-13B-code'    : "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/f6a96af4-8bf9-4294-96d6-d71aa787612e",
    'llama-34B-code'    : "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/df2bee43-fb69-42b9-9ee5-f4eabbeaf3a8",
    'llama-2-13B-chat'  : "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/e0bb7fb9-5333-4a27-8534-c6288f921d3f",
    'llama-2-70B-chat'  : "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/0e349b44-440a-44e1-93e9-abe8dcb27158",
    'mistral-7B-inst'   : "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/35ec3354-2681-4d0e-a8dd-80325dcf7c63",
    'nemotron-2-8B-QA'  : "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/0c60f14d-46cb-465e-b994-227e1c3d5047",
    'nemotron-SteerLM'  : "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/1423ff2f-d1c7-4061-82a7-9e8c67afd43a",
    # 'neva-22b'          : "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/8bf70738-59b9-4e5f-bc87-7ab4203be7a0",
    # 'fuyu-8b'           : "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/9f757064-657f-4c85-abd7-37a7a9b6ee11",
}

class AIPlayClientArgs(BaseModel):
    """
    Arguments for interacting with the AI Playground API.
    These arguments are validated both on construction and at call.
    """
    
    temperature: float = Field( 0.2, le=1, ge=0)
    top_p:       float = Field( 0.7, le=1, ge=0)
    max_tokens:  float = Field(1024, le=1024, ge=32)
    stream:      bool  = Field(False)
    max_tries:   int   = Field(5, ge=1)

    inputs: Any = Field([])
    labels: Optional[Dict[str,float]] = Field({})
    stop: Optional[Union[Sequence[str], str]] = Field([])

    arg_keys: Sequence[str]
    values: Optional[dict]

    @root_validator()
    def validate_model(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        get_str_list = lambda v: [v] if (isinstance(v, str) or not hasattr(v, '__iter__')) else v
        values['stop'] = get_str_list(values.get('stop', []))
        values['inputs'] = get_str_list(values.get('inputs'))
        values['values'] = values
        return values

    def __call__(self, *args, **kwargs):
        named_args = {k:v for k,v in zip(self.arg_keys, args)}
        named_args = {**self.values, **named_args, **kwargs}
        return AIPlayClientArgs(**named_args)


## Get a key from https://catalog.stg.ngc.nvidia.com/orgs/nvidia/models/llama2-70b/api or similar
## If running as standalone file, feel free to insert key here via os.environ['NVAPI_KEY'] = 'nvapi-...'
# os.environ['NVAPI_KEY'] = 'nvapi-...'

class AIPlayClient(BaseModel):
    """
    Underlying Client for interacting with the AI Playground API.
    Leveraged by the AIPlayBaseModel to provide a simple requests-oriented interface.
    Direct abstraction over NGC-recommended streaming/non-streaming Python solutions.

    NOTE: AI Playground does not currently support raw text continuation.
    TODO: Add support for raw text continuation for arbitrary (non-AIP) nvcf functions.
    """
                
    nv_apikey:        str      = ""
    invoke_url:       str      = ""
    fetch_url_format: str      = "https://stg.api.nvcf.nvidia.com/v2/nvcf/pexec/status/"
    get_session_fn:   Callable = requests.Session
    get_asession_fn:  Callable = aiohttp.ClientSession

    call_args_model: Optional[AIPlayClientArgs]

    arg_keys : List[str] = Field(['inputs', 'labels', 'stop'])
    gen_keys : List[str] = Field(['temperature', 'top_p', 'max_tokens', 'stream'])
    valid_roles : List[str] = Field(['user', 'system', 'assistant', 'context'])

    ## Status Tracking Variables
    last_payload  : Optional[dict] = None
    last_inputs   : Optional[dict] = None
    last_response : Optional[Any]  = None
    last_stats    : Optional[dict] = {}

    headers : Dict[str, Dict[str, str]] = dict(
        call = {
            "Authorization": "Bearer {nv_apikey}",
            "Accept": "application/json"
        },
        stream = {
            "Authorization": "Bearer {nv_apikey}",
            "Accept": "text/event-stream",
            "content-type": "application/json"
        },
    )


    stream_headers : dict  = {
        "Authorization": "Bearer {nv_apikey}",
        "Accept": "text/event-stream",
        "content-type": "application/json"
    }

    @property
    def lc_secrets(self) -> Dict[str, str]:
        """Keeps secrets out of the serialization results"""
        return {"nv_apikey": "NV_APIKEY"}

    ## Replaces __init__
    @root_validator()
    def validate_model(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        '''
        Validate that models are valid and superficially that the API key is correctly formatted.
        '''
        values['nv_apikey'] = get_from_dict_or_env(values, 'nv_apikey', 'NVAPI_KEY')
        assert 'nvapi-' in values.get('nv_apikey', ''), "[AIPlayClient Error] Invalid NVAPI key detected"
        assert values.get('invoke_url'), "[AIPlayClient Error] Invalid invoke_url detected"
        values['call_args_model'] = AIPlayClientArgs(**values)
        for header in values['headers'].values():
            if '{nv_apikey}' in header['Authorization']:
                header['Authorization'] = header['Authorization'].format(nv_apikey=values['nv_apikey'])
        return values

    ## Default Call Behavior
    def __call__(self, *args, **kwargs):
        '''
        Calls the AI Playground API with the given arguments.
        Directs to `generate` or `stream` depending on the `stream` argument.
        '''
        stream = kwargs.get('stream', kwargs.get('streaming', self.call_args_model.stream))
        out_fn = self.get_stream if stream else self.generate
        return out_fn(*args, **kwargs)

    def get_payload(self, call_args:AIPlayClientArgs):
        '''
        Generates a payload for the AI Playground API to send to service via requests.
        '''
        messages = self.preprocess(call_args)
        payvars = {k:getattr(call_args, k) for k in self.gen_keys}
        payload = dict(**messages, **payvars)
        self.last_payload = payload
        return payload

    def preprocess(self, call_args:AIPlayClientArgs) -> dict:
        '''
        Prepares a message or list of messages for the AI Playground API.
        '''
        messages = [self.prep_msg(m) for m in call_args.inputs]
        labels = call_args.labels
        if labels:
            messages += [{'labels' : labels, 'role' : 'assistant'}]
        return {'messages' : messages}


    def prep_msg(self, msg: Union[str,dict]):
        '''
        Helper Method: Ensures a message is a dictionary with a role and content.
        Example: prep_msg('Hello') -> {'role':'user', 'content':'Hello'}
        '''
        if isinstance(msg, str):
            return dict(role='user', content=msg)
        if isinstance(msg, dict):
            if msg.get('role', '') not in self.valid_roles:
                raise ValueError(f"Unknown message role \"{msg.get('role', '')}\"")
            if msg.get('content', None) is None:
                raise ValueError(f"Message {msg} has no content")
            return msg
        raise ValueError(f'Unknown message recieved: {msg}')


    def postprocess(self, response:Union[str,dict,Response], call_args:AIPlayClientArgs):
        '''
        Parses a response from the AI Playground API.
        Strongly assumes that the API will return a single response.
        '''
        if hasattr(response, 'json'):
            msg_list = [response.json()]
        elif isinstance(response, str):
            try:
                msg_list = response.split('\n\n')
                msg_list = [json.loads(response[response.find('{'):]) for response in msg_list if '{' in response]
            except JSONDecodeError:
                msg_list = []
        elif isinstance(response, dict):
            msg_list = [response]
        else:
            raise ValueError(f"Recieved ill-formed response: {response}")
        ## Dig into retrieved message and tease out ['choices'][0]['delta'/'message']
        content = ""
        is_stopped = False
        content_holder = {'content': ''}
        for msg in msg_list:
            self.last_stats = msg
            msg = msg.get('choices', [{}])[0]
            is_stopped = msg.get('finish_reason', '') == 'stop'
            msg = msg.get('delta', msg.get('message', {'content':''}))
            content_holder = msg
            content += msg.get('content', '')
            if is_stopped: break
        content_holder['content'] = content
        ## Try to early-terminate streaming or generation
        if content:
            for stopper in call_args.stop:
                if stopper and stopper in content:
                    content_holder['content'] = content[:content.find(stopper)+1]
                    is_stopped = True
        return content_holder, is_stopped

    ## Generate/Stream Options

    def generate(self, *args, **kwargs) -> dict:
        '''
        Generates a single response from the AI Playground API.
        Parses arguments, generates payload, and calls the API via `requests`.
        Strongly assumes that the API will return a single response without streaming.
        '''
        kwargs['stream'] = False
        call_args = self.call_args_model(*args, **kwargs)
        ## Construct payload, including headers, session info, url, etc.
        payload = self.get_payload(call_args)
        session = self.get_session_fn()
        ## Call the API and wait for a response, retrying if necessary
        num_tries = 0 
        self.last_inputs = dict(url=self.invoke_url, headers=self.headers['call'], json=payload, stream=False)
        self.last_response = session.post(**self.last_inputs)
        while self.last_response.status_code == 202:
            request_id = self.last_response.headers.get("NVCF-REQID")
            self.last_response = session.get(self.fetch_url_format + request_id, headers=self.headers['call'])
            num_tries += 1
            if self.last_response.status_code == 202:
                try: response_body = json.loads(self.last_response)
                except: response_body = str(self.last_response)
                if num_tries > call_args.max_tries:
                    return f"Error generating after {call_args.max_tries} attempts: response `{response_body}`"
            self.last_response.raise_for_status()
        output, _ = self.postprocess(self.last_response, call_args)
        return output


    def get_stream(self, *args, **kwargs) -> dict:
        kwargs['stream'] = True
        call_args = self.call_args_model(*args, **kwargs)
        payload = self.get_payload(call_args)
        session = self.get_session_fn()
        self.last_inputs = dict(url=self.invoke_url, headers=self.headers['stream'], json=payload, stream=True)
        self.last_response = session.post(**self.last_inputs)
        for line in self.last_response.iter_lines():
            if line and line.strip() != b"data: [DONE]":
                line = line.decode('utf-8')
                msg, final_line = self.postprocess(line, call_args)
                yield msg
                if final_line: break
            self.last_response.raise_for_status()


    async def get_astream(self, *args, **kwargs) -> dict:
        kwargs['stream'] = True
        call_args = self.call_args_model(*args, **kwargs)
        payload = self.get_payload(call_args)
        self.last_inputs = dict(url=self.invoke_url, headers=self.headers['stream'], json=payload)
        async with self.get_asession_fn() as session:
            async with session.post(**self.last_inputs) as self.last_response:
                async for line in self.last_response.content.iter_any():
                    if line and line.strip() != b"data: [DONE]":
                        line = line.decode('utf-8')
                        msg, final_line = self.postprocess(line, call_args)
                        yield msg
                        if final_line: break
                self.last_response.raise_for_status()


################################################################################

class AIPlayBaseModel(BaseModel):
    """
    Base class for NVIDIA AI Playground models which can interface with AIPlayClient.
    To be subclassed by AIPlayLLM/AIPlayChat by combining with LLM/SimpleChatModel.
    """

    client: Any = Field(AIPlayClient)
    model_name: str = Field(default="llama-2-13B-chat", description="Name of model to pull from AI_PLAY_URLS")
    model: Optional[str] = Field(description="Alias for model_name")
    model_url: Optional[str] = Field(description="URL of NVCF (NVIDIA Call Function) from NGC")
    labels: Optional[dict] = Field(description="Steer-LM labels")
    nv_apikey: Optional[str] = Field(description="NVIDIA API Key for AI Playground")
    streaming: bool = Field(False, description="Make the model generate response via streaming on call")

    @property
    def lc_secrets(self) -> Dict[str, str]:
        """Keeps secrets out of the serialization results"""
        return {"nv_apikey": "NV_APIKEY"}

    @property
    def _llm_type(self) -> str:
        """Return type of NVIDIA AI Playground Interface."""
        return "nvidia_ai_playground"

    @root_validator()
    def validate_model(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        ## Small staging/production discrepancy check
        values['nv_apikey'] = get_from_dict_or_env(values, 'nv_apikey', 'NVAPI_KEY', "")
        assert values.get('nv_apikey') and 'nvapi-' in values['nv_apikey'], "Invalid NVAPI key detected"
        is_staging = values['nv_apikey'].startswith('nvapi-stg-')
        AIP_URLs = AI_PLAY_URLS_STG if is_staging else AI_PLAY_URLS
        if values.get('model'):
            values['model_name'] = values.get('model')
        model_name = values.get('model_name')
        invoke_url = values.get('model_url')
        if model_name and not invoke_url:
            invoke_url = AIP_URLs.get(model_name)
            if not invoke_url:
                raise ValueError(f'Failed to resolve url from model name {model_name}')
        values['invoke_url'] = invoke_url
        values['client'] = values['client'](**values)
        return values

    def _call(
        self,
        messages: Union[List[BaseMessage], str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> str:
        """Simpler interface."""
        inputs = self.custom_preprocess(messages)
        labels = kwargs.get('labels', self.labels)
        if kwargs.get('streaming', self.streaming) or stop or self.client.call_args_model.stop:
            buffer = ''
            for chunk in self._stream(messages=messages, stop=stop, run_manager=run_manager, **kwargs):
                buffer += chunk if isinstance(chunk, str) else chunk.text
            responses = {'content' : buffer}
        else:
            responses = self.custom_generate(inputs, labels=labels, stop=stop, **kwargs)
        outputs = self.custom_postprocess(responses)
        return outputs

    def _get_filled_chunk(
        self,
        text: str,
        role: Optional[str] = 'assistant'
    ) -> Union[GenerationChunk, ChatGenerationChunk]:
        '''LLM and BasicChatModel have different streaming chunk specifications'''
        if isinstance(self, LLM):
            return GenerationChunk(text=text)
        return ChatGenerationChunk(message=ChatMessageChunk(content=text, role=role))

    def _stream(
        self,
        messages: Union[List[BaseMessage], str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> Iterator[Union[GenerationChunk, ChatGenerationChunk]]:
        '''Allows streaming to model just fine!'''
        inputs = self.custom_preprocess(messages)
        labels = kwargs.get('labels', self.labels)
        for response in self.custom_stream(inputs, labels=labels, stop=stop, **kwargs):
            chunk = self._get_filled_chunk(self.custom_postprocess(response))
            yield chunk
            if run_manager:
                if isinstance(run_manager, (AsyncCallbackManager, AsyncCallbackManagerForLLMRun)):
                    ## Edge case from LLM/SimpleChatModel default async methods
                    asyncio.run(run_manager.on_llm_new_token(chunk.text, chunk=chunk))
                else: 
                    run_manager.on_llm_new_token(chunk.text, chunk=chunk)
    async def _astream(
        self,
        messages: Union[List[BaseMessage], str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManager] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Union[GenerationChunk, ChatGenerationChunk]]:
        """TODO: Implement this properly. This is a lie, and merely recycles _stream..."""
        inputs = self.custom_preprocess(messages)
        labels = kwargs.get('labels', self.labels)
        async for response in await self.custom_astream(inputs, labels=labels, stop=stop, **kwargs):
            chunk = self._get_filled_chunk(self.custom_postprocess(response))
            yield chunk
            if run_manager:
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)

    def custom_preprocess(self, msgs) -> List[Dict[str,str]]:
        if isinstance(msgs, str):
            msgs = re.split("///ROLE ", msgs.strip())
            if msgs[0] == "": msgs = msgs[1:]
        elif not hasattr(msgs, '__iter__'):
            msgs = [msgs]
        out = [self.preprocess_msg(m) for m in msgs]
        return out

    def preprocess_msg(self, msg: Union[str,Sequence[str],dict,BaseMessage]) -> Dict[str,str]:
        ## Support for just simple string inputs of ///ROLE SYS etc. inputs
        if isinstance(msg, str):
            msg_split = re.split("SYS: |USER: |AGENT: |CONTEXT:", msg)
            if len(msg_split) == 1:
                return {'role':'user', 'content':msg}
            else:
                role_convert = {'AGENT':'assistant', 'SYS':'system'}
                role, content = msg.split(': ')[:2]
                role = role_convert.get(role, 'user')
                return {'role':role, 'content':content}
        ## Support for tuple inputs
        if type(msg) in (list, tuple):
            return {'role':msg[0], 'content':msg[1]}
        ## Support for manually-specified default inputs to AI Playground
        if isinstance(msg, dict) and msg.get('content'):
            msg['role'] = msg.get('role', 'user')
            return msg
        ## Support for LangChain Messages
        if hasattr(msg, 'content'):
            role_convert = {'ai':'assistant', 'system':'system'}
            role = getattr(msg, 'type')
            cont = getattr(msg, 'content')
            role = role_convert.get(role, 'user')
            if hasattr(msg, 'role'):
                cont = f"{getattr(msg, 'role')}: {cont}"
            return {'role':role, 'content':cont}

        raise ValueError(f"Invalid message: {msg}")

    def custom_generate(self, msg, labels:dict, stop=[], **kwargs) -> str:
        return self.client(msg, stream=False, labels=labels, stop=stop, **kwargs)

    def custom_stream(self, msg, labels:dict, stop=[], **kwargs) -> Iterator:
        return self.client.get_stream(msg, labels=labels, stop=stop, **kwargs)

    async def custom_astream(self, msg, labels:dict, stop=[], **kwargs) -> AsyncIterator:
        return self.client.get_astream(msg, labels=labels, stop=stop, **kwargs)

    def custom_postprocess(self, msg) -> str:
        try:
            return msg['content']
        except:
            logger.warning(f'Got ambiguous message in postprocessing, so returning as-is: msg = {msg}')
            return msg


################################################################################

class AIPlayLLM(AIPlayBaseModel, LLM):
    pass

if STANDALONE:
    class AIPlayChat(AIPlayBaseModel, SimpleChatModel):
        pass

################################################################################

class LlamaLLM(AIPlayLLM):
    model_name = Field(default="llama-13B-code")

class MistralLLM(AIPlayLLM):
    model_name = Field(default="mistral-7B-inst")

class SteerLM(AIPlayLLM):
    model_name = Field(default="nemotron-SteerLM")
    labels = Field(default={
        "creativity": 5,
        "helpfulness": 5,
        "humor": 5,
        "quality": 5
    })

class NemotronQA(AIPlayLLM):
    model_name = Field(default="nemotron-2-8B-QA")    

if STANDALONE:
    class LlamaChat(AIPlayChat):
        model_name = Field(default='llama-2-13B-chat')

    class MistralChat(AIPlayChat):
        model_name = Field(default="mistral-7B-inst")

    class SteerLMChat(AIPlayChat):
        model_name = Field(default="nemotron-SteerLM")
        labels = Field(default={
            "creativity": 5,
            "helpfulness": 5,
            "humor": 5,
            "quality": 5
        })

    class NemotronQAChat(AIPlayChat):
        model_name = Field(default="nemotron-2-8B-QA")    

