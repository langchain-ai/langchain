from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from langchain_core.output_parsers import BaseOutputParser


class GuardrailsOutputParser(BaseOutputParser):
    """Parse the output of an LLM call using Guardrails."""

    guard: Any
    """The Guardrails object."""
    api: Optional[Callable]
    """The LLM API passed to Guardrails during parsing. An example is `openai.completions.create`."""  # noqa: E501
    args: Any
    """Positional arguments to pass to the above LLM API callable."""
    kwargs: Any
    """Keyword arguments to pass to the above LLM API callable."""

    @property
    def _type(self) -> str:
        return "guardrails"

    @classmethod
    def from_rail(
        cls,
        rail_file: str,
        num_reasks: int = 1,
        api: Optional[Callable] = None,
        *args: Any,
        **kwargs: Any,
    ) -> GuardrailsOutputParser:
        """Create a GuardrailsOutputParser from a rail file.

        Args:
            rail_file: a rail file.
            num_reasks: number of times to re-ask the question.
            api: the API to use for the Guardrails object.
            *args: The arguments to pass to the API
            **kwargs: The keyword arguments to pass to the API.

        Returns:
            GuardrailsOutputParser
        """
        try:
            from guardrails import Guard
        except ImportError:
            raise ImportError(
                "guardrails-ai package not installed. "
                "Install it by running `pip install guardrails-ai`."
            )
        return cls(
            guard=Guard.from_rail(rail_file, num_reasks=num_reasks),
            api=api,
            args=args,
            kwargs=kwargs,
        )

    @classmethod
    def from_rail_string(
        cls,
        rail_str: str,
        num_reasks: int = 1,
        api: Optional[Callable] = None,
        *args: Any,
        **kwargs: Any,
    ) -> GuardrailsOutputParser:
        try:
            from guardrails import Guard
        except ImportError:
            raise ImportError(
                "guardrails-ai package not installed. "
                "Install it by running `pip install guardrails-ai`."
            )
        return cls(
            guard=Guard.from_rail_string(rail_str, num_reasks=num_reasks),
            api=api,
            args=args,
            kwargs=kwargs,
        )

    @classmethod
    def from_pydantic(
        cls,
        output_class: Any,
        num_reasks: int = 1,
        api: Optional[Callable] = None,
        *args: Any,
        **kwargs: Any,
    ) -> GuardrailsOutputParser:
        try:
            from guardrails import Guard
        except ImportError:
            raise ImportError(
                "guardrails-ai package not installed. "
                "Install it by running `pip install guardrails-ai`."
            )
        return cls(
            guard=Guard.from_pydantic(output_class, "", num_reasks=num_reasks),
            api=api,
            args=args,
            kwargs=kwargs,
        )

    def get_format_instructions(self) -> str:
        return self.guard.raw_prompt.format_instructions

    def parse(self, text: str) -> Dict:
        return self.guard.parse(text, llm_api=self.api, *self.args, **self.kwargs)
