"""Wrapper around MiniMaxChatCompletion API."""
import logging
import requests
from typing import Any, Dict, List, Mapping, Optional

from pydantic import Extra, Field, root_validator

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


class MiniMaxChatCompletion(LLM):
    """Wrapper around Minimax large language models.

    To use, you should have the environment variable ``MINIMAX_GROUP_ID`` and
    ``MINIMAX_API_KEY`` set with your API token, or pass it as a named parameter to
    the constructor.

    Example:
        .. code-block:: python

            from langchain.llms import MiniMaxChatCompletion
            llm = MiniMaxChatCompletion(model_name="abab5-chat")

    """

    client: Any

    model_name: str = "abab5-chat"
    """Model name to use"""

    temperature: float = 0.9
    """What sampling temperature to use"""

    tokens_to_generate: int = 256
    """The maximum number of tokens to generate in the completion.
    -1 returns as many tokens as possible given the prompt and
    the models maximal context size."""

    top_p: float = 0.95
    """Total probability mass of tokens to consider at each step."""

    minimax_group_id: Optional[str] = None
    """Group ID for MiniMax API."""

    minimax_api_key: Optional[str] = None
    """API Key for MiniMax API."""

    class Config:
        """Configuration for this pydantic config."""

        extra = Extra.ignore

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["minimax_group_id"] = get_from_dict_or_env(
            values, "minimax_group_id", "MINIMAX_GROUP_ID"
        )
        values["minimax_api_key"] = get_from_dict_or_env(
            values, "minimax_api_key", "MINIMAX_API_KEY"
        )
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling GooseAI API."""
        normal_params = {
            "top_p": self.top_p,
            "temperature": self.temperature,            
            "tokens_to_generate": self.tokens_to_generate,
        }
        return {**normal_params}

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_name": self.model_name}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "minimax"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the MiniMax API."""
        headers = {
            "Authorization": f"Bearer {self.minimax_api_key}",
            "Content-Type": "application/json"
        }
        url = f"https://api.minimax.chat/v1/text/chatcompletion?GroupId={self.minimax_group_id}"
        payload = {
            "model": "abab5-chat",
            "messages": [
                {
                    "sender_type": "USER",
                    "text": prompt
                }
            ],
            "tokens_to_generate": self.tokens_to_generate,
            "temperature": self.temperature,
            "top_p": self.top_p
        }
        response = requests.post(url, headers=headers, json=payload)
        parsed_response = response.json()
        base_resp = parsed_response['base_resp']
        if base_resp['status_code'] != 0:
            logger.error(base_resp['status_code'])
            raise Exception(
                "Post model outputs failed, status: "
                + base_resp['status_msg']
            )
        text = parsed_response['reply']
        
        if stop is not None:
            # This is required since the stop tokens are not enforced by the model parameters
            text = enforce_stop_tokens(text, stop)
        return text
