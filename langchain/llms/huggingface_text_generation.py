from typing import Any, Dict, List, Optional

from pydantic import root_validator, Field

from langchain.llms.base import LLM
from langchain.utils import get_from_dict_or_env


class HFTextGeneration(LLM):
    repo_id: str
    token: str
    streaming: bool = False
    model_kwargs: dict = Field(default_factory=dict)

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        huggingfacehub_api_token = get_from_dict_or_env(
            values, "huggingfacehub_api_token", "HUGGINGFACEHUB_API_TOKEN"
        )
        try:
            from text_generation import InferenceAPIClient

        except ImportError:
            raise ValueError(
                "Could not import huggingface_hub python package. "
                "Please it install it with `pip install huggingface_hub`."
            )
        values["token"] = huggingfacehub_api_token
        return values

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        from text_generation import InferenceAPIClient
        client = InferenceAPIClient(
            repo_id=self.repo_id,
            token=self.token,
        )
        if not self.streaming:
            return client.generate(prompt, **self.model_kwargs).generated_text
        if self.streaming:
            text = ""
            for response in client.generate_stream(prompt, **self.model_kwargs):
                if not response.token.special:
                    self.callback_manager.on_llm_new_token(
                        response.token.text, verbose=self.verbose
                    )
                    text += response.token.text
            return text

    async def _acall(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        from text_generation import InferenceAPIAsyncClient
        client = InferenceAPIAsyncClient(
            repo_id=self.repo_id,
            token=self.token,
        )
        if not self.streaming:
            response = await client.generate(prompt, **self.model_kwargs)
            return response.generated_text
        if self.streaming:
            text = ""
            async for response in client.generate_stream(prompt, **self.model_kwargs):
                if not response.token.special:
                    await self.callback_manager.on_llm_new_token(
                        response.token.text, verbose=self.verbose
                    )
                    text += response.token.text
            return text

    @property
    def _llm_type(self) -> str:
        raise NotImplementedError
