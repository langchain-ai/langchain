import os
import logging
import time
from typing import Any, Dict, List, Mapping, Optional, Iterator, Union, AsyncIterator
import asyncio
from concurrent.futures import ThreadPoolExecutor

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
from langchain_core.pydantic_v1 import Extra, Field, root_validator, SecretStr

logger = logging.getLogger(__name__)

def run_in_executor(executor, func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(executor, func, *args, **kwargs)

def _convert_message_to_dict(message: BaseMessage) -> dict:
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    elif isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        return {"role": "assistant", "content": message.content}
    else:
        raise ValueError(f"Got unknown type {message}")

def _convert_dict_to_message(message_dict: Mapping[str, Any]) -> BaseMessage:
    role = message_dict["role"]
    content = message_dict.get("content", "")
    if role == "user":
        return HumanMessage(content=content)
    elif role == "assistant":
        return AIMessage(content=content)
    elif role == "system":
        return SystemMessage(content=content)
    else:
        raise ValueError(f"Got unknown role {role}")

class ChatGithub(BaseChatModel):
    """GitHub LLM with Azure Fallback"""

    github_endpoint_url: str = "https://models.inference.ai.azure.com/chat/completions"
    model: str
    github_api_key: SecretStr = Field(default_factory=lambda: SecretStr(os.environ.get("GITHUB_TOKEN", "")))
    azure_api_key: SecretStr = Field(default_factory=lambda: SecretStr(os.environ.get("AZURE_API_KEY", "")))
    system_prompt: Optional[str] = None
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    use_azure_fallback: bool = True
    rate_limit_reset_time: float = 0
    request_count: int = 0
    max_requests_per_minute: int = 15
    max_requests_per_day: int = 150
    streaming: bool = False

    SUPPORTED_MODELS = [
        "AI21-Jamba-Instruct",
        "cohere-command-r",
        "cohere-command-r-plus",
        "cohere-embed-v3-english",
        "cohere-embed-v3-multilingual",
        "meta-llama-3-70b-instruct",
        "meta-llama-3-8b-instruct",
        "meta-llama-3.1-405b-instruct",
        "meta-llama-3.1-70b-instruct",
        "meta-llama-3.1-8b-instruct",
        "mistral-large",
        "mistral-large-2407",
        "mistral-nemo",
        "mistral-small",
        "gpt-4o",
        "gpt-4o-mini",
        "phi-3-medium-instruct-128k",
        "phi-3-medium-instruct-4k",
        "phi-3-mini-instruct-128k",
        "phi-3-mini-instruct-4k",
        "phi-3-small-instruct-128k",
        "phi-3-small-instruct-8k"
    ]

    MODEL_TOKEN_LIMITS = {
        "AI21-Jamba-Instruct": {"input": 72000, "output": 4000},
        "cohere-command-r": {"input": 131000, "output": 4000},
        "cohere-command-r-plus": {"input": 131000, "output": 4000},
        "meta-llama-3-70b-instruct": {"input": 8000, "output": 4000},
        "meta-llama-3-8b-instruct": {"input": 8000, "output": 4000},
        "meta-llama-3.1-405b-instruct": {"input": 131000, "output": 4000},
        "meta-llama-3.1-70b-instruct": {"input": 131000, "output": 4000},
        "meta-llama-3.1-8b-instruct": {"input": 131000, "output": 4000},
        "mistral-large": {"input": 33000, "output": 4000},
        "mistral-large-2407": {"input": 131000, "output": 4000},
        "mistral-nemo": {"input": 131000, "output": 4000},
        "mistral-small": {"input": 33000, "output": 4000},
        "gpt-4o": {"input": 131000, "output": 4000},
        "gpt-4o-mini": {"input": 131000, "output": 4000},
        "phi-3-medium-instruct-128k": {"input": 131000, "output": 4000},
        "phi-3-medium-instruct-4k": {"input": 4000, "output": 4000},
        "phi-3-mini-instruct-128k": {"input": 131000, "output": 4000},
        "phi-3-mini-instruct-4k": {"input": 4000, "output": 4000},
        "phi-3-small-instruct-128k": {"input": 131000, "output": 4000},
        "phi-3-small-instruct-8k": {"input": 131000, "output": 4000},
    }

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api keys are set."""
        github_api_key = values.get("github_api_key") or os.environ.get("GITHUB_TOKEN")
        azure_api_key = values.get("azure_api_key") or os.environ.get("AZURE_API_KEY")

        if not github_api_key:
            raise ValueError("GITHUB_TOKEN must be set in the environment or passed in.")
        if not azure_api_key and values.get("use_azure_fallback"):
            raise ValueError("AZURE_API_KEY must be set for Azure fallback.")

        return values
    
    def _prepare_messages(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Prepare messages for API call, including system prompt if present."""
        message_dicts = []
        
        if self.system_prompt:
            message_dicts.append({"role": "system", "content": self.system_prompt})
        
        message_dicts.extend([_convert_message_to_dict(m) for m in messages])
        
        return message_dicts

    def _count_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Estimate token count for messages. This is a simplified method and may not be exact."""
        return sum(len(m['content'].split()) for m in messages) * 1.3  # Guestimates the Token Count
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "github_with_azure_fallback"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "github_endpoint_url": self.github_endpoint_url,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "use_azure_fallback": self.use_azure_fallback,
            **{"model_kwargs": self.model_kwargs},
        }

    def _check_rate_limit(self) -> bool:
        """Check if the rate limit has been reached."""
        current_time = time.time()
        if current_time < self.rate_limit_reset_time:
            return False
        if self.request_count >= self.max_requests_per_minute:
            self.rate_limit_reset_time = current_time + 60
            self.request_count = 0
            return False
        return True

    def _increment_request_count(self):
        """Increment the request count."""
        self.request_count += 1

    def _call_api(self, endpoint_url: str, headers: Dict[str, str], data: Dict[str, Any], stream: bool = False) -> Any:
        """Make an API call to either GitHub or Azure."""
        if stream:
            response = requests.post(endpoint_url, headers=headers, json=data, stream=True)
        else:
            response = requests.post(endpoint_url, headers=headers, json=data)
        response.raise_for_status()
        return response

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            return self._stream_generate(messages, stop, run_manager, **kwargs)

        if not self._check_token_limit(message_dicts):
            raise ValueError(f"Input tokens exceed the maximum limit for model {self.model}")

        message_dicts = self._prepare_messages(messages)
        data = {
            "messages": message_dicts,
            "model": self.model,
            **self.model_kwargs,
            **kwargs
        }

        if self._check_rate_limit():
            try:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.github_api_key.get_secret_value()}"
                }
                response = self._call_api(self.github_endpoint_url, headers, data)
                self._increment_request_count()
                content = response.json()["choices"][0]["message"]["content"]
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])
            except (requests.exceptions.RequestException, ValueError) as e:
                logger.warning(f"GitHub API call failed: {str(e)}. Falling back to Azure.")

        if self.use_azure_fallback:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.azure_api_key.get_secret_value()}"
            }
            response = self._call_api(self.github_endpoint_url, headers, data)
            content = response.json()["choices"][0]["message"]["content"]
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])
        else:
            raise ValueError("Rate limit reached and Azure fallback is disabled.")

    def _stream_generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        message_dicts = self._prepare_messages(messages)
        
        if not self._check_token_limit(message_dicts):
            raise ValueError(f"Input tokens exceed the maximum limit for model {self.model}")

        data = {
            "messages": message_dicts,
            "model": self.model,
            "stream": True,
            **self.model_kwargs,
            **kwargs
        }

        if self._check_rate_limit():
            try:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.github_api_key.get_secret_value()}"
                }
                response = self._call_api(self.github_endpoint_url, headers, data, stream=True)
                self._increment_request_count()
                return self._process_streaming_response(response, run_manager)
            except (requests.exceptions.RequestException, ValueError) as e:
                logger.warning(f"GitHub API call failed: {str(e)}. Falling back to Azure.")

        if self.use_azure_fallback:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.azure_api_key.get_secret_value()}"
            }
            response = self._call_api(self.github_endpoint_url, headers, data, stream=True)
            return self._process_streaming_response(response, run_manager)
        else:
            raise ValueError("Rate limit reached and Azure fallback is disabled.")

    def _process_streaming_response(
        self,
        response: requests.Response,
        run_manager: Optional[CallbackManagerForLLMRun],
    ) -> ChatResult:
        content = ""
        for line in response.iter_lines():
            if line:
                chunk = line.decode('utf-8')
                content += chunk
                if run_manager:
                    run_manager.on_llm_new_token(chunk)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return await run_in_executor(
            None, self._generate, messages, stop, run_manager, **kwargs
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts = self._prepare_messages(messages)
        data = {
            "messages": message_dicts,
            "model": self.model,
            "stream": True,
            **self.model_kwargs,
            **kwargs
        }

        if self._check_rate_limit():
            try:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.github_api_key.get_secret_value()}"
                }
                response = self._call_api(self.github_endpoint_url, headers, data, stream=True)
                self._increment_request_count()
                for line in response.iter_lines():
                    if line:
                        chunk = line.decode('utf-8')
                        yield ChatGenerationChunk(message=AIMessage(content=chunk))
                        if run_manager:
                            run_manager.on_llm_new_token(chunk)
                return
            except (requests.exceptions.RequestException, ValueError) as e:
                logger.warning(f"GitHub API call failed: {str(e)}. Falling back to Azure.")

        if self.use_azure_fallback:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.azure_api_key.get_secret_value()}"
            }
            response = self._call_api(self.github_endpoint_url, headers, data, stream=True)
            for line in response.iter_lines():
                if line:
                    chunk = line.decode('utf-8')
                    yield ChatGenerationChunk(message=AIMessage(content=chunk))
                    if run_manager:
                        run_manager.on_llm_new_token(chunk)
        else:
            raise ValueError("Rate limit reached and Azure fallback is disabled.")

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        async for chunk in self._async_stream_generator(messages, stop, run_manager, **kwargs):
            yield chunk

    async def _async_stream_generator(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        for chunk in self._stream(messages, stop, run_manager, **kwargs):
            yield chunk
            await asyncio.sleep(0)
