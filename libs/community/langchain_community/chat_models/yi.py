"""Wrapper around Yi chat models."""

from typing import Dict, Any, Optional

from langchain_core.utils import (
    convert_to_secret_str,
    get_from_dict_or_env,
    pre_init,
)

from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms.yi import YiCommon, YI_SERVICE_URL_DOMESTIC, YI_SERVICE_URL_INTERNATIONAL

class YiChat(YiCommon, ChatOpenAI):  # type: ignore[misc]
    """Yi large language models.

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``YI_API_KEY`` set with your API key.

    Referenced from https://platform.01.ai/docs

    Referenced from https://platform.lingyiwanwu.com/docs

    Example:
        .. code-block:: python

            from langchain_community.chat_models.yi import YiChat

            yi_chat = YiChat(model="yi-large")
    """

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the environment is set up correctly."""
        values["yi_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(values, "yi_api_key", "YI_API_KEY")
        )

        try:
            import openai
        except ImportError:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )

        base_url = values.get("base_url", YI_SERVICE_URL_INTERNATIONAL)
        if base_url not in [YI_SERVICE_URL_DOMESTIC, YI_SERVICE_URL_INTERNATIONAL]:
            raise ValueError("Invalid base_url. Must be either domestic or international Yi API URL.")

        client_params = {
            "api_key": values["yi_api_key"].get_secret_value(),
            "base_url": base_url,
        }

        if not values.get("client"):
            values["client"] = openai.OpenAI(**client_params).chat.completions
        if not values.get("async_client"):
            values["async_client"] = openai.AsyncOpenAI(
                **client_params
            ).chat.completions

        return values

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "yi-chat"

    def _get_invocation_params(self, **kwargs: Any) -> Dict[str, Any]:
        """Get the parameters used to invoke the model."""
        params = super()._get_invocation_params(**kwargs)
        # Ensure we're using the correct model name
        params["model"] = self.model_name
        return params

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_name": self.model_name}, **super()._identifying_params}

if __name__ == '__main__':
    os.environ["YI_API_KEY"] = "70116f6e1e2947dda1d75c18c4f59280"
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(
            content="你是一个翻译官，你的任务就是将英语翻译成中文"
        ),
        HumanMessage(
            content="我爱你"
        ),
    ]
    yi_chat = YiChat()
    print(yi_chat(messages))
