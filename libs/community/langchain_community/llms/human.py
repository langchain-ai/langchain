from typing import Any, Callable, List, Mapping, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from pydantic import Field

from langchain_community.llms.utils import enforce_stop_tokens


def _display_prompt(prompt: str) -> None:
    """Displays the given prompt to the user."""
    print(f"\n{prompt}")  # noqa: T201


def _collect_user_input(
    separator: Optional[str] = None, stop: Optional[List[str]] = None
) -> str:
    """Collects and returns user input as a single string."""
    separator = separator or "\n"
    lines = []

    while True:
        line = input()
        if not line:
            break
        lines.append(line)

        if stop and any(seq in line for seq in stop):
            break
    # Combine all lines into a single string
    multi_line_input = separator.join(lines)
    return multi_line_input


class HumanInputLLM(LLM):
    """User input as the response."""

    input_func: Callable = Field(default_factory=lambda: _collect_user_input)
    prompt_func: Callable[[str], None] = Field(default_factory=lambda: _display_prompt)
    separator: str = "\n"
    input_kwargs: Mapping[str, Any] = {}
    prompt_kwargs: Mapping[str, Any] = {}

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """
        Returns an empty dictionary as there are no identifying parameters.
        """
        return {}

    @property
    def _llm_type(self) -> str:
        """Returns the type of LLM."""
        return "human-input"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Displays the prompt to the user and returns their input as a response.

        Args:
            prompt (str): The prompt to be displayed to the user.
            stop (Optional[List[str]]): A list of stop strings.
            run_manager (Optional[CallbackManagerForLLMRun]): Currently not used.

        Returns:
            str: The user's input as a response.
        """
        self.prompt_func(prompt, **self.prompt_kwargs)
        user_input = self.input_func(
            separator=self.separator, stop=stop, **self.input_kwargs
        )

        if stop is not None:
            # I believe this is required since the stop tokens
            # are not enforced by the human themselves
            user_input = enforce_stop_tokens(user_input, stop)
        return user_input
