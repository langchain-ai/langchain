import os
import logging
import time
from typing import Any, Dict, List, Mapping, Optional, Generator

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Extra, Field, root_validator

logger = logging.getLogger(__name__)

class GithubLLM(LLM):
    """GitHub LLM with Azure Fallback

    This module allows using LLMs hosted on GitHub's inference endpoint with automatic fallback to Azure when rate limits are reached.

    To use this module, you must:
    * Export your GitHub token as the environment variable `GITHUB_TOKEN`
    * Export your Azure API key as the environment variable `AZURE_API_KEY` (for fallback)
    * Specify the model name you want to use

    Example:
        .. code-block:: python

        from langchain_community.llms import GithubLLM

        # Make sure GITHUB_TOKEN and AZURE_API_KEY are set in your environment variables
        llm = GithubLLM(model="gpt-4o", system_prompt="You are a knowledgeable history teacher.", use_azure_fallback=True)

        # Single turn conversation
        response = llm("What is the capital of France?")
        print(response)

        # Multi-turn conversation
        conversation = [
            {"role": "user", "content": "Tell me about the French Revolution."},
            {"role": "assistant", "content": "The French Revolution was a period of major social and political upheaval in France that began in 1789 with the Storming of the Bastille and ended in the late 1790s with the ascent of Napoleon Bonaparte. It was partially carried forward by Napoleon during the later expansion of the French Empire. The Revolution overthrew the monarchy, established a republic, catalyzed violent periods of political turmoil, and fundamentally altered French history."},
            {"role": "user", "content": "What were the main causes?"}
        ]

        response = llm.chat(conversation)
        print(response)

        # Streaming with chat history
        for chunk in llm.stream("Can you elaborate on the Reign of Terror?", chat_history=conversation):
            print(chunk, end='', flush=True)

        # This will raise a ValueError
        # llm_invalid = GithubLLM(model="invalid-model")
    """

    github_endpoint_url: str = "https://models.inference.ai.azure.com/chat/completions"
    model: str
    system_prompt: str = "You are a helpful assistant."
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    use_azure_fallback: bool = True
    rate_limit_reset_time: float = 0
    request_count: int = 0
    max_requests_per_minute: int = 15  # Adjust based on your tier
    max_requests_per_day: int = 150  # Adjust based on your tier

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

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    @root_validator("model")
    def validate_model(cls, v):
        if v.lower() not in [model.lower() for model in cls.SUPPORTED_MODELS]:
            raise ValueError(f"Model {v} is not supported. Please choose from {cls.SUPPORTED_MODELS}")
        return v

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = {field.alias for field in cls.__fields__.values()}

        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name not in all_required_field_names:
                if field_name in extra:
                    raise ValueError(f"Found {field_name} supplied twice.")
                logger.warning(
                    f"""{field_name} was transferred to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)
        values["model_kwargs"] = extra
        return values

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

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "github_with_azure_fallback"

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

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> str:
        """Call the LLM with fallback to Azure if rate limited."""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        if chat_history:
            messages.extend(chat_history)
        
        messages.append({"role": "user", "content": prompt})

        data = {
            "messages": messages,
            "model": self.model,
            **self.model_kwargs,
            **kwargs
        }

        if self._check_rate_limit():
            try:
                github_token = os.environ.get("GITHUB_TOKEN")
                if not github_token:
                    raise ValueError("GITHUB_TOKEN environment variable is not set.")

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {github_token}"
                }

                response = self._call_api(self.github_endpoint_url, headers, data)
                self._increment_request_count()
                return response.json()["choices"][0]["message"]["content"]
            except (requests.exceptions.RequestException, ValueError) as e:
                logger.warning(f"GitHub API call failed: {str(e)}. Falling back to Azure.")

        if self.use_azure_fallback:
            azure_api_key = os.environ.get("AZURE_API_KEY")
            if not azure_api_key:
                raise ValueError("AZURE_API_KEY environment variable is not set.")

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {azure_api_key}"
            }

            response = self._call_api(self.github_endpoint_url, headers, data)
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise ValueError("Rate limit reached and Azure fallback is disabled.")

    def stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """Stream the response from the LLM with fallback to Azure if rate limited."""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        if chat_history:
            messages.extend(chat_history)
        
        messages.append({"role": "user", "content": prompt})

        data = {
            "messages": messages,
            "model": self.model,
            "stream": True,
            **self.model_kwargs,
            **kwargs
        }

        if self._check_rate_limit():
            try:
                github_token = os.environ.get("GITHUB_TOKEN")
                if not github_token:
                    raise ValueError("GITHUB_TOKEN environment variable is not set.")

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {github_token}"
                }

                response = self._call_api(self.github_endpoint_url, headers, data, stream=True)
                self._increment_request_count()
                for line in response.iter_lines():
                    if line:
                        yield line.decode('utf-8')
                return
            except (requests.exceptions.RequestException, ValueError) as e:
                logger.warning(f"GitHub API call failed: {str(e)}. Falling back to Azure.")

        if self.use_azure_fallback:
            azure_api_key = os.environ.get("AZURE_API_KEY")
            if not azure_api_key:
                raise ValueError("AZURE_API_KEY environment variable is not set.")

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {azure_api_key}"
            }

            response = self._call_api(self.github_endpoint_url, headers, data, stream=True)
            for line in response.iter_lines():
                if line:
                    yield line.decode('utf-8')
        else:
            raise ValueError("Rate limit reached and Azure fallback is disabled.")

    def chat(
        self,
        messages: List[Dict[str, str]],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Conduct a multi-turn conversation with the LLM."""
        return self._call(
            prompt=messages[-1]["content"],
            chat_history=messages[:-1],
            stop=stop,
            run_manager=run_manager,
            **kwargs
        )