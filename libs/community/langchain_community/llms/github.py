import os
import logging
from typing import Any, Dict, List, Mapping, Optional

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Extra, Field, root_validator

logger = logging.getLogger(__name__)

class GithubLLM(LLM):
    """GitHub LLM

    This module allows using LLMs hosted on GitHub's inference endpoint.

    To use this module, you must:
    * Export your GitHub token as the environment variable `GITHUB_TOKEN`
    * Specify the model name you want to use

    Example:
        .. code-block:: python

        from langchain_community.llms import GithubLLM

        # Make sure GITHUB_TOKEN is set in your environment variables
        llm = GithubLLM(model="gpt-4o", system_prompt="You are a knowledgeable history teacher.")

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
    """

    endpoint_url: str = "https://models.inference.ai.azure.com/chat/completions"
    model: str
    system_prompt: str = "You are a helpful assistant."
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

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
            "endpoint_url": self.endpoint_url,
            "model": self.model,
            "system_prompt": self.system_prompt,
            **{"model_kwargs": self.model_kwargs},
        }

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "github"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> str:
        """Call the GitHub LLM."""
        github_token = os.environ.get("GITHUB_TOKEN")
        if not github_token:
            raise ValueError("GITHUB_TOKEN environment variable is not set.")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {github_token}"
        }

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

        response = requests.post(self.endpoint_url, headers=headers, json=data)
        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"]

    def stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> str:
        """Stream the response from the GitHub LLM."""
        github_token = os.environ.get("GITHUB_TOKEN")
        if not github_token:
            raise ValueError("GITHUB_TOKEN environment variable is not set.")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {github_token}"
        }

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

        response = requests.post(self.endpoint_url, headers=headers, json=data, stream=True)
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                yield line.decode('utf-8')

    def chat(
        self,
        messages: List[Dict[str, str]],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Conduct a multi-turn conversation with the GitHub LLM."""
        return self._call(
            prompt=messages[-1]["content"],
            chat_history=messages[:-1],
            stop=stop,
            run_manager=run_manager,
            **kwargs
        )