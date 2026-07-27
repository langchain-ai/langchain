"""LLM factory: create LangChain-compatible chat model instances based on configuration.

This module isolates provider-specific creation so agents depend only on an abstract
chat model interface (LangChain chat models). Changing provider is a config change.
"""
from __future__ import annotations

from typing import Any

"""LLM factory: create simple async LLM wrappers based on configuration.

This module provides a minimal async wrapper around the OpenAI ChatCompletion
API to keep dependencies light and make the LLM invocation explicit and testable.
Other providers (Anthropic, Ollama) can be added following the same pattern.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict

import openai

from content_growth_agent.config.settings import settings


class AsyncOpenAIWrapper:
    """Minimal async wrapper around OpenAI ChatCompletion.

    Provides `agenerate` which accepts a prompt string and returns the assistant reply.
    """

    def __init__(self, model: str, api_key: str, temperature: float = 0.2, use_langchain: bool = False) -> None:
        if not api_key:
            raise RuntimeError("OpenAI API key is not set. Set OPENAI_API_KEY in .env")
        openai.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.use_langchain = use_langchain

        # Optional LangChain ChatOpenAI integration
        self._langchain_model = None
        if use_langchain:
            try:
                from langchain.chat_models import ChatOpenAI

                self._langchain_model = ChatOpenAI(model=self.model, temperature=self.temperature, openai_api_key=api_key)
            except Exception:
                # If LangChain is not available or fails, fall back to direct OpenAI
                self._langchain_model = None

    async def agenerate(self, prompt: str, **kwargs: Any) -> str:
        """Call the LLM asynchronously and return assistant text."""
        if self._langchain_model is not None:
            # Use LangChain model via thread to avoid blocking
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._sync_langchain_call, prompt, kwargs)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync_call, prompt, kwargs)

    def _sync_langchain_call(self, prompt: str, kwargs: Dict[str, Any]) -> str:
        # LangChain ChatOpenAI has a `generate`/`predict` API; use `predict` if available
        try:
            # prefer simple predict if present
            if hasattr(self._langchain_model, "predict"):
                return self._langchain_model.predict(prompt)
            # else try generate -> messages handling
            resp = self._langchain_model.generate([{"role": "user", "content": prompt}])
            # resp.generations is nested; extract first
            gens = getattr(resp, "generations", None)
            if gens:
                first = gens[0][0]
                return getattr(first, "text", str(first))
        except Exception:
            pass
        return ""

    def _sync_call(self, prompt: str, kwargs: Dict[str, Any]) -> str:
        messages = [
            {"role": "system", "content": kwargs.get("system", "")},
            {"role": "user", "content": prompt},
        ]
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            **kwargs,
        )
        # Get assistant text
        choices = response.get("choices", [])
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "")


def get_llm(**override: Any) -> AsyncOpenAIWrapper:
    """Instantiate and return an async LLM wrapper based on settings/overrides.

    Example override: model='gpt-4', temperature=0.3
    """
    provider = override.get("provider", settings.llm_provider.value)
    model = override.get("model", settings.llm_model)
    temperature = override.get("temperature", 0.2)
    use_langchain = override.get("use_langchain", True)

    if provider == "openai":
        api_key = override.get("openai_api_key", settings.openai_api_key)
        return AsyncOpenAIWrapper(model=model, api_key=api_key, temperature=temperature, use_langchain=use_langchain)

    raise NotImplementedError(f"LLM provider '{provider}' is not implemented yet")
