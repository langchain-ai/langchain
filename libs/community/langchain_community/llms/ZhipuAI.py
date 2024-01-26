import logging
from typing import Any, List, Mapping, Optional

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain.llms.base import LLM

from langchain.llms.utils import enforce_stop_tokens

logger = logging.getLogger(__name__)


class ZhipuAI(LLM):
    """ZhipuAI LLM service.

    Example:
        . code-block:: python

            from langchain_community.llms import ZhipuAI
            ZhipuAI_llm = ZhipuAI(model_name="glm-4", ZhipuAI_api_key="")
    """

    model_kwargs: Optional[dict] = None
    """Keyword arguments to pass to the model."""
    max_token: int = 4096
    """Max token allowed to pass to the model.1-4096"""
    temperature: float = 0.1
    """LLM model temperature from 0 to 10."""
    top_p: float = 0.7
    """Top P for nucleus sampling from 0 to 1"""
    model_name: str = "glm-4"
    """Name of model to use"""
    ZhipuAI_api_key = ""
    """ZhipuAI API key"""

    @property
    def _llm_type(self) -> str:
        return "ZhipuAI"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"model_kwargs": _model_kwargs}
        }

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        """
        Example:
            . code-block:: python

                response = zhipu_llm("Who are you?")
        """

        _model_kwargs = self.model_kwargs or {}

        # call api
        try:
            from zhipuai import ZhipuAI as zp
            client = zp(api_key=self.ZhipuAI_api_key)
            response = client.chat.completions.create(
                model="glm-4",  # 填写需要调用的模型名称
                messages=[
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_token,
                temperature=self.temperature,
                top_p=self.top_p
            )
        except ImportError:
            raise ImportError(
                "Could not import zhipuai python package. "
                "Please it install it with `pip install zhipuai`."
            )
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error raised by inference endpoint: {e}")

        logger.debug(f"ZhipuAI response: {response}")


        try:
            parsed_response = response.choices[0].message.content

            # Check if response content does exists
            if parsed_response:
                text = parsed_response
            else:
                raise ValueError(f"Unexpected response type: {parsed_response}")

        except requests.exceptions.JSONDecodeError as e:
            raise ValueError(
                f"Error raised during decoding response from inference endpoint: {e}."
                f"\nResponse: {response}"
            )

        if stop is not None:
            text = enforce_stop_tokens(text, stop)
        return text
