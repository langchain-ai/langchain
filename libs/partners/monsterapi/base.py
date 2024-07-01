from typing import Any, Callable, Dict, Optional, Sequence

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.types import BaseOutputParser, PydanticProgramMode

DEFAULT_MONSTER_TEMP = 0.75


class MonsterAPIIntegration:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url

    def generate_text(self, model: str, prompt: str, temperature: float, max_length: int) -> Dict[str, Any]:
        endpoint = f"{self.base_url}/models/{model}/generate"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "prompt": prompt,
            "temperature": temperature,
            "max_length": max_length,
        }

        try:
            response = requests.post(endpoint, json=data, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error generating text: {e}")
            return {"error": str(e)}


class MonsterLLM(CustomLLM):
    model: str = Field(description="The MonsterAPI model to use.")
    monster_api_key: Optional[str] = Field(description="The MonsterAPI key to use.")
    max_new_tokens: int = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="The number of tokens to generate.",
        gt=0,
    )
    temperature: float = Field(
        default=DEFAULT_MONSTER_TEMP,
        description="The temperature to use for sampling.",
        gte=0.0,
        lte=1.0,
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The number of context tokens available to the LLM.",
        gt=0,
    )

    _client: Any = PrivateAttr()

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.monsterapi.ai/v1",
        monster_api_key: Optional[str] = None,
        max_new_tokens: int = DEFAULT_NUM_OUTPUTS,
        temperature: float = DEFAULT_MONSTER_TEMP,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        self._client = MonsterAPIIntegration(monster_api_key, base_url)

        super().__init__(
            model=model,
            monster_api_key=monster_api_key,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            context_window=context_window,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_new_tokens,
            model_name=self.model,
        )

    def complete(
        self, prompt: str, formatted: bool = False, timeout: int = 100, **kwargs: Any
    ) -> CompletionResponse:
        if not formatted:
            prompt = self.completion_to_prompt(prompt)

        input_dict = {
            "prompt": prompt,
            "temperature": self.temperature,
            "max_length": self.max_new_tokens,
            **kwargs,
        }

        result = self._client.generate_text(
            model=self.model,
            prompt=prompt,
            temperature=self.temperature,
            max_length=self.max_new_tokens,
        )

        if "error" in result:
            raise RuntimeError(result["error"])

        return CompletionResponse(text=result["text"])

    # Add other methods as needed for streaming or specific callbacks

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = self.messages_to_prompt(messages)
        return self.complete(prompt, formatted=True, **kwargs)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        # Implement streaming logic if supported by MonsterAPI
        pass
