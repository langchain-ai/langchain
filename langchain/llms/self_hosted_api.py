import logging
from typing import Optional, List, Any, Dict, Mapping, Type, Union, Tuple
from pydantic import Extra, root_validator, BaseModel, Field
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.utils import enforce_stop_tokens
import requests

logger = logging.getLogger(__name__)

VALID_TASKS = ('text-generation', 
               'summarization', 
               'translation', 
               'question-answering', 
               'fill-mask', 
               'zero-shot-classification', 
               'text-classification', 
               'text2text-generation', 
               )

class SelfHostedApi(LLM):
    """
    Wrapper around self-hosted large language models exposed via web API

    To use, you should have the ``requests`` python package installed.

    Supports any task for which the input can effectively be sent in json format. Currently, this includes
    a multitude of text manipulation/classification tasks, but not image or audio tasks.

    Example using an api expecting input like ``{"prompt": "This is a prompt"}`` and
    returning output like ``{"response": "This is a response"}``:
        .. code-block:: python

            from langchain.llms import SelfHostedApi
            llm = SelfHostedApi(endpoint_url="http://localhost:8000", 
                                task="text-generation")

    Example using an api expecting a different input schema and returning a different
    output schema:
        .. code-block:: python
            from langchain.llms import SelfHostedApi
            from pydantic import BaseModel

            class InputSchema(BaseModel):
                input_prompt: str
                max_length: int = 50
                temperature: float = 0.9
            
            class OutputSchema(BaseModel):
                generated_response: str
                initial_prompt: str
                time_to_inference: float

            llm = SelfHostedApi(endpoint_url="http://localhost:8000",
                                task="text-generation",
                                input_schema=InputSchema,
                                output_schema=OutputSchema,
                                prompt_key="input_prompt",
                                response_key="generated_response")
            

    """
    endpoint_url: str 
    """The url of the target endpoint (e.g. http://localhost:8000, https://myapi.com/prompt)"""
    task: str
    """The task to be performed by the API. Must be in VALID_TASKS"""
    model_kwargs: Optional[Dict] = None 
    """Holds any parameters that the api accepts that you want to exist for the lifetime of the LLM"""
    input_schema: Optional[Type[BaseModel]] = None
    """The class representing the input schema the API is expecting. Must be a pydantic BaseModel"""
    output_schema: Optional[Type[BaseModel]] = None
    """The class representing the output schema the API is returning. Must be a pydantic BaseModel"""
    prompt_key: Optional[str] = 'prompt'
    """The key in the input schema that the API expects the prompt to be under"""
    response_key: Optional[Union[int, str, Tuple[Union[str, int], ...]]] = 'response'
    """The key in the output schema that the API returns the response under"""

    class Config:
        """Config for pydantic model"""
        extra = Extra.forbid


    @root_validator
    def prompt_key_in_input_schema(cls, values):
        """Validator ensuring the prompt key is in the input schema (if it exists)"""
        prompt_key = values.get('prompt_key')
        input_schema = values.get('input_schema')
        if input_schema is None:
            return values
        elif prompt_key not in input_schema.schema()['properties']:
            raise ValueError("Prompt key must be in input schema")
        else:
            return values

    @root_validator
    def response_key_in_output_schema(cls, values):
        # TODO: Verify nested key existence in schema at instance creation time
        
        return values

    @root_validator
    def model_kwargs_in_input_schema(cls, values):
        """Validator ensuring the keys in model kwargs are present in the input schema"""
        model_kwargs = values.get('model_kwargs')
        input_schema = values.get('input_schema')
        if model_kwargs is None:
            return values
        elif input_schema is None:
            raise ValueError("Model kwargs specified but no input schema")
        elif not all(k in input_schema.schema()['properties'] for k in model_kwargs):
            raise ValueError("Model kwargs must be in input schema")
        else:
            return values

    @root_validator
    def task_must_be_valid(cls, values):
        """Validator ensuring the task is valid"""
        task = values.get('task')
        if task not in VALID_TASKS:
            raise ValueError(f"Task must be one of {VALID_TASKS}")
        else:
            return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "endpoint_url": self.endpoint_url,
            # "request_type": self.request_type,
            "input_schema": self.input_schema.schema() if self.input_schema is not None else None,
            "output_schema": self.output_schema.schema() if self.output_schema is not None else None,
            'prompt_key': self.prompt_key ,
            'response_key': self.response_key,
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "self_hosted_api"

    def _call(
        self,
        prompt:str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the API with the prompt and return the response"""

        if kwargs:
            if not all(k in self.input_schema.schema()['properties'] for k in kwargs):
                raise ValueError("kwargs must be in input schema")

        model_kwargs = self.model_kwargs or {}
        if self.input_schema is not None:
            if self.input_schema.schema()['properties'][self.prompt_key]['type'] == 'string':
                input_model = self.input_schema(**{self.prompt_key: prompt, **model_kwargs, **kwargs}) 
                request_body = input_model.dict()
            elif self.input_schema.schema()['properties'][self.prompt_key]['type'] == 'array':
                input_model = self.input_schema(**{self.prompt_key: [prompt], **model_kwargs, **kwargs}) 
                request_body = input_model.dict()
            else:
                raise ValueError('Invalid type for prompt in input schema. Must be string or array type (e.g. List, Tuple)')
        else:
            request_body = {self.prompt_key: prompt}

        try:
            response = requests.post(self.endpoint_url, json=request_body)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error calling endpoint: {e}")

        response_body = response.json() 

        if isinstance(self.response_key, (str, int)):
            self.response_key = (self.response_key,)
        try:
            for layer in self.response_key:
                response_body = response_body[layer]
            response_text = response_body

        except KeyError:
            raise ValueError(f"Response key {layer} not found in response body: {response_body}")
        except IndexError:
            raise ValueError(f"Response index {layer} not found in response body: {response_body}")
        except Exception as e:
            raise ValueError(f"Error parsing response body: {e}")

        if stop is not None:
            response_text = enforce_stop_tokens(response_text, stop)
        return response_text
        