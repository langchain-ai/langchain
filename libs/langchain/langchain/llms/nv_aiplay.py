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

from typing import Callable, Any, Dict, List, Optional, Tuple, Sequence, Union, AsyncIterator, Iterator
from requests.models import Response

import logging
logger = logging.getLogger(__name__)

class ClientModel(BaseModel):
    '''
    Custom BaseModel subclass with some desirable properties for subclassing
    '''
    def subscope(self, *args, **kwargs):
        '''Create a new ClientModel with the same values but new arguments'''
        named_args = {k:v for k,v in zip(getattr(self, 'arg_keys', []), args)}
        named_args = {**named_args, **kwargs}
        out = self.copy(update=named_args)
        for k,v in self.__dict__.items():
            if isinstance(v, ClientModel):
                setattr(out, k, v.copy(update=named_args))
        return out

    def get(self, key:str):
        '''Get a value from the ClientModel, using it like a dictionary'''
        return getattr(self, key)

    def transfer_state(self, other):
        '''Transfer state from one ClientModel to another'''
        for k,v in self.__dict__.items():
            if k in getattr(self, 'state_vars', []):
                setattr(other, k, v)
            elif hasattr(v, 'transfer_state'):
                other_sub = getattr(other, k, None)
                if other_sub is not None:
                    v.transfer_state(other_sub)


class NVCRModel(ClientModel):

    """
    Underlying Client for interacting with the AI Playground API.
    Leveraged by the NVAIPlayBaseModel to provide a simple requests-oriented interface.
    Direct abstraction over NGC-recommended streaming/non-streaming Python solutions.

    NOTE: AI Playground does not currently support raw text continuation.
    TODO: Add support for raw text continuation for arbitrary (non-AIP) nvcf functions.
    """

    ## Core defaults. These probably should not be changed
    fetch_url_format: str = Field("https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/")
    call_invoke_base: str = Field("https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions")
    get_session_fn: Callable = Field(requests.Session)
    get_asession_fn: Callable = Field(aiohttp.ClientSession)

    ## Populated on construction/validation
    nvapi_key: Optional[str]
    is_staging: Optional[bool]
    available_models: Optional[Dict[str, str]] 

    ## Generation arguments
    max_tries: int = Field(5, ge=1)
    stop: Optional[Union[str, List[str]]]
    headers = dict(
        call = {
            "Authorization": "Bearer {nvapi_key}",
            "Accept": "application/json"
        },
        stream = {
            "Authorization": "Bearer {nvapi_key}",
            "Accept": "text/event-stream",
            "content-type": "application/json"
        },
    )

    ## Status Tracking Variables. Updated Progressively
    last_inputs   : dict = None
    last_response : Any  = None
    last_msg      : dict = {}
    state_vars: Sequence[str] = ['last_inputs', 'last_response', 'last_msg']        

    @root_validator()
    def validate_model(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        '''Validate and update model arguments, including API key and formatting'''
        values['nvapi_key'] = get_from_dict_or_env(values, 'nvapi_key', 'NVAPI_KEY')
        if 'nvapi-' not in values.get('nvapi_key', ''):
            raise ValueError("Invalid NVAPI key detected. Should start with `nvapi-`")
        values['is_staging'] = 'nvapi-stg-' in values['nvapi_key']
        for header in values['headers'].values():
            if '{nvapi_key}' in header['Authorization']:
                header['Authorization'] = header['Authorization'].format(nvapi_key=values['nvapi_key'])
        if isinstance(values['stop'], str):
            values['stop'] = [values['stop']]
        return values

    def __init__(self, *args, **kwargs):
        '''Useful to define custom operations on construction after validation'''
        super().__init__(*args, **kwargs)
        self.fetch_url_format = self._stagify(self.fetch_url_format)
        self.call_invoke_base = self._stagify(self.call_invoke_base)
        try:
            self.available_models = self.get_available_models()
        except Exception as e:
            raise Exception("Error retrieving model list. Verify your NVAPI key") from e

    def _stagify(self, path):
        '''Helper method to switch between staging and production endpoints'''
        if self.is_staging and 'stg.api' not in path: 
            return path.replace('api', 'stg.api')
        if not self.is_staging and 'stg.api' in path: 
            return path.replace('stg.api', 'api')
        return path
        
    ####################################################################################
    ## Core utilities for posting and getting from NVCR

    def _post(self, invoke_url:str, payload:dict={}) -> Tuple[Response, Any]:
        '''Method for posting to the AI Playground API.'''
        self.last_inputs = dict(url=invoke_url, headers=self.headers['call'], json=payload, stream=False)
        session = self.get_session_fn()
        self.last_response = session.post(**self.last_inputs)
        return self.last_response, session

    def _get(self, invoke_url:str, payload:dict={}) -> Tuple[Response, Any]:
        '''Method for getting from the AI Playground API.'''
        self.last_inputs = dict(url=invoke_url, headers=self.headers['call'], json=payload, stream=False)
        session = self.get_session_fn()
        self.last_response = session.get(**self.last_inputs)
        return self.last_response, session

    def _wait(self, response:Response, session:Any) -> Response:
        '''Method for waiting for a response from the AI Playground API after an initial response is made.'''
        num_tries = 0 
        while response.status_code == 202:
            request_id = response.headers.get("NVCF-REQID")
            response = session.get(self.fetch_url_format + request_id, headers=self.headers['call'])
            num_tries += 1
            if response.status_code == 202:
                try: response_body = json.loads(response)
                except: response_body = str(response)
                if num_tries > self.max_tries:
                    return f"Error generating after {self.max_tries} attempts: response `{response_body}`"
            response.raise_for_status()
        return response        
    
    ####################################################################################
    ## Simple query interface to show the set of model options

    def query(self, invoke_url:str, payload:dict={}) -> dict:
        '''Simple-as-possible method for an end-to-end get query. Returns result dictionary'''
        response, session = self._get(invoke_url, payload)
        response = self._wait(response, session)
        output = self._process_response(response)[0]
        return output

    def _process_response(self, response:Union[str,Response]) -> List[dict]:
        '''General-purpose response processing for single responses and streams'''
        if hasattr(response, 'json'):      ## For single response (i.e. non-streaming)
            try: return [response.json()]
            except json.JSONDecodeError: pass
        elif isinstance(response, str):    ## For set of responses (i.e. streaming)
            msg_list = []
            for msg in response.split('\n\n'):
                if '{' not in msg: continue
                msg_list += [json.loads(msg[msg.find('{'):])]
            return msg_list
        raise ValueError(f"Recieved ill-formed response: {response}")

    def get_available_models(self) -> dict:
        '''Get a dictionary of available models from the AI Playground API.'''
        invoke_url = self._stagify("https://api.nvcf.nvidia.com/v2/nvcf/functions")
        return {v['name'] : v['id'] for v in self.query(invoke_url)['functions']}

    def _get_invoke_url(
        self,
        model_name:Optional[str]=None, 
        invoke_url:Optional[str]=None
    ) -> str:
        '''Helper method to get the invoke URL from a model name, URL, or endpoint stub'''
        if not invoke_url:           
            if not model_name:
                raise ValueError("URL or model name must be specified to invoke")
            if model_name in self.available_models:
                invoke_url = self.available_models.get(model_name)
            else: 
                for k,v in self.available_models.items():
                    if model_name in k:
                        invoke_url = v
                        break
        if not invoke_url: 
            raise ValueError(f'Unknown model name {model_name} specified')
        if 'http' not in invoke_url:
            invoke_url = f'{self.call_invoke_base}/{invoke_url}'
        return invoke_url

    ####################################################################################
    ## Generation interface to allow users to generate new values from endpoints

    def get_req_generation(
        self, 
        model_name:Optional[str]=None, 
        payload:dict={}, 
        invoke_url:Optional[str]=None
    ) -> dict:
        '''Method for an end-to-end post query with NVCR post-processing.'''
        invoke_url = self._get_invoke_url(model_name, invoke_url)
        if payload.get('stream', False) == True:
            payload = {**payload, 'stream' : False}
        response, session = self._post(invoke_url, payload)
        response = self._wait(response, session)
        output, _ = self.postprocess(response)
        return output

    def postprocess(self, response:Union[str,Response]) -> Tuple[dict, bool]:
        '''Parses a response from the AI Playground API.
        Strongly assumes that the API will return a single response.
        '''
        msg_list = self._process_response(response)
        msg, is_stopped = self._aggregate_msgs(msg_list)
        msg, is_stopped = self._early_stop_msg(msg, is_stopped)
        return msg, is_stopped

    def _aggregate_msgs(self, msg_list:List[dict]) -> Tuple[dict, bool]:
        '''Dig into retrieved message and tease out ['choices'][0]['delta'/'message']'''
        content_buffer = ''
        content_holder = {'content': ''}
        is_stopped = False
        for msg in msg_list:
            self.last_msg = msg
            msg = msg.get('choices', [{}])[0]
            is_stopped = msg.get('finish_reason', '') == 'stop'
            msg = msg.get('delta', msg.get('message', {'content':''}))
            content_holder = msg
            content_buffer += msg.get('content', '')
            if is_stopped: break
        content_holder['content'] = content_buffer
        return content_holder, is_stopped

    def _early_stop_msg(self, msg:dict, is_stopped:bool) -> Tuple[dict, bool]:
        '''Try to early-terminate streaming or generation by iterating over stop list'''
        content = msg.get('content', '')
        if content and self.stop: 
            for stop_str in self.stop:
                if stop_str and stop_str in content:
                    msg['content'] = content[:content.find(stop_str)+1]
                    is_stopped = True
        return msg, is_stopped

    ####################################################################################
    ## Streaming interface to allow you to iterate through progressive generations

    def get_req_stream(
        self, 
        model:Optional[str]=None, 
        payload:dict={}, 
        invoke_url:Optional[str]=None
    ) -> Iterator:
        invoke_url = self._get_invoke_url(model, invoke_url)
        if payload.get('stream', True) == False:
            payload = {**payload, 'stream' : True}
        self.last_inputs = dict(url=invoke_url, headers=self.headers['stream'], json=payload, stream=True)
        self.last_response = self.get_session_fn().post(**self.last_inputs)
        for line in self.last_response.iter_lines():
            if line and line.strip() != b"data: [DONE]":
                line = line.decode('utf-8')
                msg, final_line = self.postprocess(line)
                yield msg
                if final_line: break
            self.last_response.raise_for_status()

    ####################################################################################
    ## Asynchronous streaming interface to allow multiple generations to happen at once.

    async def get_req_astream(
        self, 
        model:Optional[str]=None, 
        payload:dict={}, 
        invoke_url:Optional[str]=None
    ) -> AsyncIterator:
        invoke_url = self._get_invoke_url(model, invoke_url)
        if payload.get('stream', True) == False:
            payload = {**payload, 'stream' : True}
        self.last_inputs = dict(url=invoke_url, headers=self.headers['stream'], json=payload)
        async with self.get_asession_fn() as session:
            async with session.post(**self.last_inputs) as self.last_response:
                async for line in self.last_response.content.iter_any():
                    if line and line.strip() != b"data: [DONE]":
                        line = line.decode('utf-8')
                        msg, final_line = self.postprocess(line)
                        yield msg
                        if final_line: break
                self.last_response.raise_for_status()


class NVAIPlayClient(ClientModel):
    '''
    Higher-Level Client for interacting with the AI Playground API with argument defaults.
    Will be subclassed by NVAIPlayLLM/NVAIPlayChat to provide a simple LangChain interface.
    '''
    
    client: Optional[ClientModel]

    ## These default can get updated in child classes. Watch out for double-fielding
    model_name: str = Field('llama2_13b')
    # model: Optional[str]
    labels: dict = Field({})

    temperature: float = Field( 0.2, le=1, gt=0)
    top_p:       float = Field( 0.7, le=1, gt=0)
    max_tokens:  int   = Field(1024, le=1024, ge=32)
    streaming:   bool  = Field(False)

    inputs: Union[Sequence[str], str] = Field([])
    stop:   Union[Sequence[str], str] = Field([])

    gen_keys:    Sequence[str] = Field(['temperature', 'top_p', 'max_tokens', 'streaming'])
    arg_keys:    Sequence[str] = Field(['inputs', 'labels', 'stop'])
    valid_roles: Sequence[str] = Field(['user', 'system', 'assistant', 'context'])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @root_validator()
    def validate_model(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values.get('client') is None:
            values['client'] = NVCRModel(**values)
        else: 
            values['client'] = values['client'].subscope(**values)
        if values.get('model'):
            values['model_name'] = values.get('model')
        values['model'] = values['model_name']
        return values

    @property
    def available_models(self) -> List[str]:
        return list(self.client.available_models.keys())

    # ## Default Call Behavior. Great for standalone use, but not for LangChain
    # def __call__(self, *args, **kwargs):
    #     '''
    #     Calls the AI Playground API with the given arguments.
    #     Directs to `generate` or `stream` depending on the `stream` argument.
    #     '''
    #     stream = kwargs.get('stream', kwargs.get('streaming', self.streaming))
    #     out_fn = self.get_stream if stream else self.get_generation
    #     return out_fn(*args, **kwargs)

    def get_generation(self, *args, **kwargs) -> dict:
        '''Call to client generate method with call scope'''
        call = self.subscope(*args, **kwargs)
        out = call.client.get_req_generation(call.model_name, call.get_payload(stream=False))
        call.transfer_state(self)
        return out
    
    def get_stream(self, *args, **kwargs) -> Iterator:
        '''Call to client stream method with call scope'''
        call = self.subscope(*args, **kwargs)
        out = call.client.get_req_stream(call.model_name, payload=call.get_payload(stream=True))
        call.transfer_state(self)
        return out

    def get_astream(self, *args, **kwargs) -> AsyncIterator:
        '''Call to client astream method with call scope'''
        call = self.subscope(*args, **kwargs)
        out = call.client.get_req_astream(call.model_name, call.get_payload(stream=True))
        call.transfer_state(self)
        return out

    def get_payload(self, *args, **kwargs) -> dict:
        '''Generates payload for the NVAIPlayClient API to send to service.'''
        k_map = lambda k: k if k != 'streaming' else 'stream' 
        out = {
            **self.preprocess(),
            **{k_map(k) : self.get(k) for k in self.gen_keys}
        }
        return out

    def preprocess(self) -> dict:
        '''Prepares a message or list of messages for the payload'''
        get_str_list = lambda v: [v] if (
            isinstance(v, str) or 
            not hasattr(v, '__iter__') or 
            isinstance(v, BaseMessage)
        ) else v
        self.inputs = get_str_list(self.inputs)
        messages = [self.prep_msg(m) for m in self.inputs]
        labels = self.labels
        if labels:
            messages += [{'labels' : labels, 'role' : 'assistant'}]
        return {'messages' : messages}

    def prep_msg(self, msg:Union[str,dict]):
        '''Helper Method: Ensures a message is a dictionary with a role and content.'''
        if isinstance(msg, str):
            return dict(role='user', content=msg)
        if isinstance(msg, dict):
            if msg.get('role', '') not in self.valid_roles:
                raise ValueError(f"Unknown message role \"{msg.get('role', '')}\"")
            if msg.get('content', None) is None:
                raise ValueError(f"Message {msg} has no content")
            return msg
        raise ValueError(f'Unknown message recieved: {msg} of type {type(msg)}')
    

class NVAIPlayBaseModel(NVAIPlayClient):
    """
    Base class for NVIDIA AI Playground models which can interface with NVAIPlayClient.
    To be subclassed by NVAIPlayLLM/NVAIPlayChat by combining with LLM/SimpleChatModel.
    """

    @property
    def _llm_type(self) -> str:
        """Return type of NVIDIA AI Playground Interface."""
        return "nvidia_ai_playground"

    def _call(
        self,
        messages: Union[List[BaseMessage], str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> str:
        '''_call hook for LLM/SimpleChatModel. Allows for streaming and non-streaming calls'''
        inputs = self.custom_preprocess(messages)
        labels = kwargs.get('labels', self.labels)
        if kwargs.get('streaming', self.streaming) or stop or self.client.stop:
            buffer = ''
            for chunk in self._stream(messages=messages, stop=stop, run_manager=run_manager, **kwargs):
                buffer += chunk if isinstance(chunk, str) else chunk.text
            responses = {'content' : buffer}
        else:
            responses = self.get_generation(inputs, labels=labels, stop=stop, **kwargs)
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
        '''Allows streaming to model!'''
        inputs = self.custom_preprocess(messages)
        labels = kwargs.get('labels', self.labels)
        for response in self.get_stream(inputs, labels=labels, stop=stop, **kwargs):
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
        inputs = self.custom_preprocess(messages)
        labels = kwargs.get('labels', self.labels)
        async for response in self.get_astream(inputs, labels=labels, stop=stop, **kwargs):
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
        raise ValueError(f"Invalid message: {repr(msg)} of type {type(msg)}")

    def custom_postprocess(self, msg) -> str:
        try:
            return msg['content']
        except:
            logger.warning(f'Got ambiguous message in postprocessing, so returning as-is: msg = {msg}')
            return msg

################################################################################

class NVAIPlayLLM(NVAIPlayBaseModel, LLM):
    pass

if STANDALONE:
    class NVAIPlayChat(NVAIPlayBaseModel, SimpleChatModel):
        pass

################################################################################

class LlamaLLM(NVAIPlayLLM):
    model_name : str = Field("llama2_code_13b", alias='model')

class MistralLLM(NVAIPlayLLM):
    model_name : str = Field("mistral", alias='model')

class SteerLM(NVAIPlayLLM):
    model_name : str = Field("gpt_steerlm_8b", alias='model')
    labels = Field({
        "creativity": 5,
        "helpfulness": 5,
        "humor": 5,
        "quality": 5
    })

class NemotronQA(NVAIPlayLLM):
    model_name : str = Field("gpt_qa_8b", alias='model')    

if STANDALONE:
    class LlamaChat(NVAIPlayChat):
        model_name : str = Field('llama2_13b', alias='model')

    class MistralChat(NVAIPlayChat):
        model_name : str = Field("mistral", alias='model')

    class SteerLMChat(NVAIPlayChat):
        model_name : str = Field("gpt_steerlm_8b", alias='model')
        labels = Field(default={
            "creativity": 5,
            "helpfulness": 5,
            "humor": 5,
            "quality": 5
        })

    class NemotronQAChat(NVAIPlayChat):
        model_name : str = Field("gpt_qa_8b", alias='model')    
