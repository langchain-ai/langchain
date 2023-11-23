from langchain.pydantic_v1 import BaseModel, Field, root_validator, validator
from langchain.schema.output import GenerationChunk, ChatGenerationChunk
from langchain.schema.messages import BaseMessage, BaseMessageChunk, ChatMessageChunk
from langchain.callbacks.manager import CallbackManager, AsyncCallbackManager
from langchain.utils import get_from_dict_or_env

try:    ## if running as part of package
    from .base import LLM   
except: ## if running as standalone file (remember to uncomment chat variants)
    from langchain.llms.base import LLM  
    from langchain.chat_models.base import SimpleChatModel  

from typing import Callable, Any, Dict, List, Optional, Tuple
import requests
import json
from json.decoder import JSONDecodeError
import os
import re

# import aiohttp
import asyncio

## Important for NeVA/Fuyu
import imghdr
import base64


# ## TODO: Pull these in from ngc if possible
# ## Staging Options
AI_PLAY_URLS_STG = {
    'llama-13B-code'    : "https://stg.api.nvcf.nvidia.com/v2/nvcf/pexec/functions/eb1100de-60bf-4e9a-8617-b7d4652e0c37",
    'llama-34B-code'    : "https://stg.api.nvcf.nvidia.com/v2/nvcf/pexec/functions/1b739bf1-f3b0-4f04-9ace-3a1cbcefca49",
    'llama-2-13B-chat'  : "https://stg.api.nvcf.nvidia.com/v2/nvcf/pexec/functions/4e38f372-2533-4e76-b3d2-8bfe7b9ca783",
    'llama-2-70B-chat'  : "https://stg.api.nvcf.nvidia.com/v2/nvcf/pexec/functions/ee2643e0-bda6-4ddc-9841-378a11f1dd04",
    'mistral-7B-inst'   : "https://stg.api.nvcf.nvidia.com/v2/nvcf/pexec/functions/33809adf-b2d7-497b-aa21-83e3f334a953",
    'neva-22b'          : "https://stg.api.nvcf.nvidia.com/v2/nvcf/pexec/functions/7c94dd3e-72d7-48de-b0fe-aaceb5aa0c23",
    'fuyu-8b'           : "https://stg.api.nvcf.nvidia.com/v2/nvcf/pexec/functions/8f0f7c3e-7e31-40f1-abe6-d3d7c526fb7b"
}

## Production Options
AI_PLAY_URLS = {
    'llama-13B-code'    : "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/f6a96af4-8bf9-4294-96d6-d71aa787612e",
    'llama-34B-code'    : "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/df2bee43-fb69-42b9-9ee5-f4eabbeaf3a8",
    'llama-2-13B-chat'  : "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/e0bb7fb9-5333-4a27-8534-c6288f921d3f",
    'llama-2-70B-chat'  : "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/0e349b44-440a-44e1-93e9-abe8dcb27158",
    'mistral-7B-inst'   : "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/35ec3354-2681-4d0e-a8dd-80325dcf7c63",
    'nemotron-2-8B-QA'  : "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/0c60f14d-46cb-465e-b994-227e1c3d5047",
    'neva-22b'          : "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/8bf70738-59b9-4e5f-bc87-7ab4203be7a0",
    'fuyu-8b'           : "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/9f757064-657f-4c85-abd7-37a7a9b6ee11"
}

# Get a key from https://catalog.stg.ngc.nvidia.com/orgs/nvidia/models/llama2-70b/api or similar

class AIPlayClient(BaseModel):

    """
    Underlying Client for interacting with the AI Playground API.
    Leveraged by the AIPlayBaseModel to provide a simple requests-oriented interface.
    Direct abstraction over NGC-recommended streaming/non-streaming Python solutions.
    """

    nv_apikey:        str      = ""
    invoke_url:       str      = ""
    fetch_url_format: str      = "https://stg.api.nvcf.nvidia.com/v2/nvcf/pexec/status/"
    get_session_fn:   Callable = requests.Session

    temperature:      float    = Field( 0.2, le=1, ge=0)
    top_p:            float    = Field( 0.7, le=1, ge=0)
    max_tokens:       float    = Field(1024, le=4096, ge=1)
    stream:           bool     = Field(False)

    max_tries:        int      = Field(3, ge=1)

    arg_keys = ['inputs', 'labels', 'stop']
    gen_keys = ['temperature', 'top_p', 'max_tokens', 'stream']
    allowed_roles = ['user', 'system', 'assistant', 'context']

    ## Status Tracking Variables
    last_payload  : Optional[dict] = None
    last_inputs   : Optional[dict] = None
    last_response : Optional[Any]  = None

    ## Replaces __init__
    @root_validator()
    def validate_model(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        '''
        Validate that models are valid and superficially that the API key is correctly formatted.
        '''
        values['nv_apikey'] = get_from_dict_or_env(values, 'nv_apikey', 'NVAPI_KEY')
        assert values.get('nv_apikey') and 'nvapi-' in values.get('nv_apikey'), "[AIPlayClient Error] Invalid NVAPI key detected"
        assert values.get('invoke_url'), "[AIPlayClient Error] Invalid invoke_url detected"
        values['default_dict'] = {k:values.get(k) for k in values.get('gen_keys')}
        return values

    ## Default Call Behavior
    def __call__(self, *args, **kwargs):
        '''
        Calls the AI Playground API with the given arguments.
        Directs to `generate` or `stream` depending on the `stream` argument.
        '''
        stream = kwargs.get('stream', self.stream)
        out_fn = self.get_stream if stream else self.generate
        return out_fn(*args, **kwargs)

    ## parse_args -> get_payload -> (preprocess -> (prep_msg))

    def parse_args(self, *args, **kwargs):
        '''
        Parses arguments to coallece args, kwargs, and defaults into a single dictionary.
        '''
        named_args = {k:v for k,v in zip(self.arg_keys, args)}
        named_args = {**self.default_dict, **named_args, **kwargs}
        get_str_list = lambda v: [v] if (isinstance(v, str) or not hasattr(v, '__iter__')) else v
        named_args['stop']   = get_str_list(named_args.get('stop', []))
        named_args['inputs'] = get_str_list(named_args.get('inputs'))
        return named_args

    def get_payload(self, **named_args):
        '''
        Generates a payload for the AI Playground API to send to service via requests.
        '''
        messages = self.preprocess(**named_args)
        payvars  = {k:v for k,v in named_args.items() if k in self.gen_keys}
        payload  = dict(**messages, **payvars)
        return payload

    def preprocess(self, inputs, **named_args) -> dict:
        '''
        Prepares a message or list of messages for the AI Playground API.
        '''
        messages = [self.prep_msg(m) for m in inputs]
        labels = named_args.get('labels')
        if labels:
            messages += [{'labels' : labels, 'role' : 'assistant'}]
        return {'messages' : messages}

    def prep_msg(self, msg):
        '''
        Helper Method: Ensures a message is a dictionary with a role and content.
        Example: prep_msg('Hello') -> {'role':'user', 'content':'Hello'}
        '''
        if isinstance(msg, str):
            return dict(role='user', content=msg)
        if isinstance(msg, dict):
            assert msg.get('role', '') in self.allowed_roles, f"Unknown message role \"{msg.get('role', '')}\""
            assert msg.get('content'), f"Message {msg} has no content"
            return msg
        raise ValueError(f'Unknown message recieved: {msg}')

    def postprocess(self, response, **named_args):
        '''
        Parses a response from the AI Playground API.
        Strongly assumes that the API will return a single response.
        '''
        msg_raw = ''
        if hasattr(response, 'json'):
            msg_raw = response.json()
        elif isinstance(response, str):
            try: msg_raw = json.loads(response[response.find('{'):])
            except JSONDecodeError: pass
        elif isinstance(response, dict):
            msg_raw = response
        assert msg_raw, (f"Recieved ill-formed response: {response}")
        ## Dig into retrieved message and tease out ['choices'][0]['delta'/'message']
        msg = msg_raw.get('choices', [{}])[0]
        msg = msg.get('delta', msg.get('message', False))
        assert msg, f"Expected message but recieved response {msg_raw}"
        ## Try to early-terminate streaming or generation
        is_stopped = False
        if 'content' in msg:
            text = msg.get('content', '')
            for stopper in named_args.get('stop'):
                if stopper and stopper in text:
                    msg['content'] = text[:text.find(stopper)+1]
                    is_stopped = True
        return msg, is_stopped

    ## Generate/Stream Options

    def generate(self, *args, **kwargs) -> dict:
        '''
        Generates a single response from the AI Playground API.
        Parses arguments, generates payload, and calls the API via `requests`.
        Strongly assumes that the API will return a single response without streaming.
        '''
        kwargs['stream'] = False
        named_args = self.parse_args(*args, **kwargs)
        ## Construct payload, including headers, session info, url, etc.
        payload = self.get_payload(**named_args)
        self.last_payload = payload
        session = self.get_session_fn()
        call_headers = {
            "Authorization": f"Bearer {self.nv_apikey}",
            "Accept": "application/json"
        }
        num_tries = 0    ## Call the API and wait for a response, retrying if necessary
        self.last_inputs = dict(url=self.invoke_url, headers=call_headers, json=payload, stream=False)
        response = session.post(**self.last_inputs)
        while response.status_code == 202:
            request_id = response.headers.get("NVCF-REQID")
            response = session.get(self.fetch_url_format + request_id, headers=call_headers)
            self.last_response = response
            num_tries += 1
            if response.status_code == 202:
                try: response_body = json.loads(response)
                except: response_body = ""
                if num_tries > self.max_tries:
                    return f"Error generating after {self.max_tries} attempts: response `{response_body}`"
            response.raise_for_status()

        output, _ = self.postprocess(response=response, **named_args)
        return output


    def get_stream(self, *args, **kwargs) -> dict:
        kwargs['stream'] = True
        named_args = self.parse_args(*args, **kwargs)
        ## Construct payload, including headers, session info, url, etc.
        payload = self.get_payload(**named_args)
        session = self.get_session_fn()
        call_headers = {
            "Authorization": f"Bearer {self.nv_apikey}",
            "Accept": "text/event-stream",
            "content-type": "application/json"
        }
        self.last_payload = payload   ## Call the API and wait for updates, retrying if necessary
        self.last_inputs = dict(url=self.invoke_url, headers=call_headers, json=payload, stream=True)
        response = session.post(**self.last_inputs)
        for line in response.iter_lines():
            self.last_response = response
            if line and line != b"data: [DONE]":
                line = line.decode('utf-8')
                msg, final_line = self.postprocess(response=line, **named_args)
                yield msg
                if final_line: break
            response.raise_for_status()

    # async def get_astream(self, *args, **kwargs) -> dict:
    #     kwargs['stream'] = True
    #     named_args = self.parse_args(*args, **kwargs)
    #     ## Construct payload, including headers, session info, url, etc.
    #     payload = self.get_payload(**named_args)
    #     call_headers = {
    #         "Authorization": f"Bearer {self.nv_apikey}",
    #         "Accept": "text/event-stream",
    #         "content-type": "application/json"
    #     }
    #     self.last_payload = payload   ## Call the API and wait for updates, retrying if necessary
    #     async with aiohttp.ClientSession() as session:
    #         self.last_inputs = dict(url=self.invoke_url, headers=call_headers, json=payload, stream=True)
    #         async with session.post(**self.last_inputs) as response:
    #             async for line in response.content:
    #                 if line and line.decode() != "data: [DONE]":
    #                     line = line.decode('utf-8')
    #                     msg, final_line = self.postprocess(response=line, **named_args)
    #                     yield msg
    #                     if final_line: break
    #             response.raise_for_status()

################################################################################

class AIPlayBaseModel(BaseModel):
    """Simple Chat Model Base."""

    client: Any = Field(AIPlayClient)
    model_name: str = Field(default="llama-2-13B-chat")
    n: int = Field(default=1, ge=1)
    model: Optional[str]
    model_url: Optional[str]
    labels: Optional[dict]
    nv_apikey: Optional[str]

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
        values['client'] = values['client'](nv_apikey=values.get('nv_apikey'), invoke_url=invoke_url)
        return values

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> str:
        """Simpler interface."""
        inputs = self.custom_preprocess(messages)
        labels = kwargs.get('labels', self.labels)
        responses = self.custom_generate(inputs, labels=labels, stop=stop)
        outputs = self.custom_postprocess(responses)
        return outputs

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ):
        """Simpler interface."""
        inputs = self.custom_preprocess(messages)
        labels = kwargs.get('labels', self.labels)
        if isinstance(self, LLM):
            chunk_fn = lambda text: GenerationChunk(text=text)
        else:
            chunk_fn = lambda text: ChatGenerationChunk(
                message = ChatMessageChunk(content=text, role='assistant'))

        for response in self.custom_stream(inputs, labels=labels, stop=stop):
            chunk = chunk_fn(self.custom_postprocess(response))
            yield chunk
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)

    # https://github.com/langchain-ai/langchain/blob/e584b28c54da3ef66cb44568ab1522fabbf1af75/libs/langchain/langchain/chat_models/fireworks.py#L218
    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManager] = None,
        **kwargs: Any,
    ):
        """TODO: Implement this properly. This is a lie, and merely recycles _stream..."""
        inputs = self.custom_preprocess(messages)
        labels = kwargs.get('labels', self.labels)
        if isinstance(self, LLM):
            chunk_fn = lambda text: GenerationChunk(text=text)
        else:
            chunk_fn = lambda text: ChatGenerationChunk(
                message = ChatMessageChunk(content=text, role='assistant'))

        # async for response in await self.custom_astream(inputs, labels=labels):
        #     chunk = chunk_fn(self.custom_postprocess(response))
        #     yield chunk
        #     if run_manager:
        #         await run_manager.on_llm_new_token(chunk.text, chunk=chunk)

        for response in self.custom_stream(inputs, labels=labels, stop=stop):
            chunk = chunk_fn(self.custom_postprocess(response))
            yield chunk
            if run_manager:
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)

    def custom_preprocess(self, msgs):
        if isinstance(msgs, str):
            msgs = re.split("///ROLE ", msgs.strip())
            if msgs[0] == "": msgs = msgs[1:]
        elif not hasattr(msgs, '__iter__'):
            msgs = [msgs]
        out = [self.preprocess_msg(m) for m in msgs]
        return out

    def preprocess_msg(self, msg):
        ## Support for just simple string inputs of ///ROLE SYS etc. inputs
        if isinstance(msg, str):
            msg_split = re.split("SYS: |USER: |AGENT:", msg)
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
        if isinstance(msg, dict):
            assert 'role' in msg and 'content' in msg, f"Invalid message: {msg}"
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

    def custom_generate(self, msg, labels, stop):
        return self.client(msg, stream=False, labels=labels, stop=stop)

    def custom_stream(self, msg, labels, stop=[]):
        return self.client(msg, stream=True, labels=labels, stop=stop)

    # async def custom_astream(self, msg, labels, stop=[]):
    #     return self.client.get_astream(msg, stream=True, labels=labels, stop=stop)

    def custom_postprocess(self, msg):
        try:
            return msg['content']
        except:
            print(f'[WARNING] Got ambiguous message in postprocessing, so returning as-is: msg = {msg}')
            return msg

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return 'llama'


################################################################################

class NeVAClient(AIPlayClient):
    
    '''
    Adds support for image inputs. NeVA on AI Playground is very early, so this will become outdated quickly. 
    We'll try to continue supporting it in bursts until it stablizes. 

    Example Invocation: 

    neva_client = NeVAClient()
    img_link = "<img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAIAQMAAAD+wSzIAAAABlBMVEX///+/v7+jQ3Y5AAAADklEQVQI12P4AIX8EAgALgAD/aNpbtEAAAAASUVORK5CYII==\" />"
    neva_client(f"Hi! What is in this image? {img_link}")
    '''

    max_tokens: float = Field(512, le=512, ge=1)

    def preprocess(self, inputs, **kwargs) -> dict:
        msg_list = [inputs] if isinstance(inputs, str) or not hasattr(inputs, '__iter__') else inputs
        inputs = [self.prep_msg(m) for m in msg_list]
        inputs = [self.parse_imgs(m) for m in inputs]
        labels = kwargs.get('labels')
        if labels is not None:
          inputs += [{'labels' : labels, 'role' : 'assistant'}]
        return {'messages' : inputs}

    def parse_imgs(self, msg):
        ## TODO: Implement message size check per new standards. 
        ## Current recommendation to limit input size to ~200kb, or use Assets API (tbd)
        content = msg.get('content', '')
        img_tags = re.findall(r'<img ([^>]+)>', content)
        for tag in img_tags:
            image_type = None
            if tag.startswith(('http://', 'https://')):
                image_b64, image_type = self.url_to_base64_and_type(tag)
            elif os.path.exists(tag):
                image_b64, image_type = self.local_to_base64_and_type(tag)
            if image_type:
                img_tag = f'<img src="data:image/{image_type};base64,{image_b64}" />'
                content = content.replace(f'<img {tag}>', img_tag)
        msg['content'] = content
        return msg

    @staticmethod
    def url_to_base64_and_type(url: str) -> tuple:
        """Converts an image URL to a base64 encoded string and detects its type."""
        response = requests.get(url)
        response.raise_for_status()
        image_type = imghdr.what(None, response.content)
        return base64.b64encode(response.content).decode('utf-8'), image_type

    @staticmethod
    def local_to_base64_and_type(image_path: str) -> tuple:
        """Converts a local image to a base64 encoded string and detects its type."""
        with open(image_path, "rb") as img_file:
            image_content = img_file.read()
            image_type = imghdr.what(None, image_content)
            return base64.b64encode(image_content).decode('utf-8'), image_type

################################################################################

class ChatLlamaClient(AIPlayClient):
    invoke_url = AI_PLAY_URLS['llama-2-13B-chat']

class CodeLlamaClient(AIPlayClient):
    invoke_url = AI_PLAY_URLS['llama-13B-code']

class MistralClient(AIPlayClient):
    invoke_url = AI_PLAY_URLS['mistral-7B-inst']

################################################################################

class AIPlayLLM(AIPlayBaseModel, LLM):
    pass

# class AIPlayChat(AIPlayBaseModel, SimpleChatModel):
#     pass

################################################################################

class LlamaLLM(AIPlayLLM):
    model_name = Field(default="llama-13B-code")

class MistralLLM(AIPlayLLM):
    model_name = Field(default="mistral-7B-inst")

# class LlamaChat(AIPlayChat):
#     model_name = Field(default='llama-2-13B-chat')

# class MistralChat(AIPlayChat):
#     model_name = Field(default="mistral-7B-inst")

################################################################################

class NevaLLM(AIPlayBaseModel, LLM):
    client: Any = Field(NeVAClient)
    model_name = Field(default="neva-22b")

class FuyuLLM(AIPlayBaseModel, LLM):
    client: Any = Field(NeVAClient)
    model_name = Field(default="fuyu-8b")

# class NevaChat(AIPlayBaseModel, SimpleChatModel):
#     client: Any = Field(NeVAClient)
#     model = Field(default="neva-22b")

# class FuyuChat(AIPlayBaseModel, SimpleChatModel):
#     client: Any = Field(NeVAClient)
#     model = Field(default="fuyu-8b")

