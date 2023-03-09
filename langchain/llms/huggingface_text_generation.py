from typing import Any, Dict, List, Optional

from pydantic import root_validator

from langchain.llms.base import LLM
from langchain.utils import get_from_dict_or_env


class HFTextGeneration(LLM):
    repo_id: str
    client: Any
    streaming: bool = False

    @root_validator()
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
        repo_id = values["repo_id"]
        client = InferenceAPIClient(
            repo_id=repo_id,
            token=huggingfacehub_api_token,
        )
        values["client"] = client
        return values

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if not self.streaming:
            return self.client.generate(prompt).generated_text
        if self.streaming:
            text = ""
            for response in self.client.generate_stream(prompt):
                if not response.token.special:
                    self.callback_manager.on_llm_new_token(
                        response.token.text, verbose=self.verbose
                    )
                    text += response.token.text
            return text

    async def _acall(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        raise NotImplementedError

    @property
    def _llm_type(self) -> str:
        raise NotImplementedError
