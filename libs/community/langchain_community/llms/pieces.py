from __future__ import annotations

from typing import Any, List, Mapping, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import GenerationChunk, LLMResult
from pieces_copilot_sdk import PiecesClient

class PiecesOSLLM(BaseLLM):
    """Pieces OS language model."""

    client: PiecesClient
    model: str = "pieces_os"

    @property
    def _llm_type(self) -> str:
        return "pieces_os"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
        }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the Pieces OS model."""
        try:
            response = self.client.ask_question(
                question=prompt,
            )
            return response
        except Exception as error:
            print(f'Error asking question: {error}')
            return 'Error asking question'

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate from the Pieces OS model."""
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
            generations.append([GenerationChunk(text=text)])
        return LLMResult(generations=generations)
