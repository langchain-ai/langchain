import os
from typing import Optional
from langchain_openai import ChatOpenAI

BOCHA_API_BASE = "https://api.bocha.ai/v1"


class ChatBocha(ChatOpenAI):
    """Bocha AI chat model.

    Uses Bocha's OpenAI-compatible API endpoint.

    Example:
        .. code-block:: python

            from langchain_bocha import ChatBocha

            llm = ChatBocha(model="deepseek-v4-pro")
            response = llm.invoke("Hello!")
    """

    model_name: str = "deepseek-v4-pro"
    openai_api_base: str = BOCHA_API_BASE

    def __init__(
        self,
        model: str = "deepseek-v4-pro",
        bocha_api_key: Optional[str] = None,
        **kwargs,
    ):
        api_key = bocha_api_key or os.environ.get("BOCHA_API_KEY", "dummy-key")
        super().__init__(
            model=model,
            openai_api_key=api_key,
            openai_api_base=BOCHA_API_BASE,
            **kwargs,
        )
