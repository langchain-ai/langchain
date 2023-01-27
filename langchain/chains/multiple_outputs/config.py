"""Configuration for variables in GetMultipleOutputsChain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Type, TypeVar

VC = TypeVar("VC", bound="VariableConfig")


@dataclass
class VariableConfig:
    """Configuration for each variable the LLM should fill in."""

    display: str
    """The string that the LLM sees as an invitation to fill this variable in.

    For example, if the value of this is "Action Input: \"", then the LLM will see the
    prompt

    ```
    Decide on your next course of action:

    Action Type: "Search"
    Action Input: "
    ```
    """
    output_key: str
    """The key in the final output dict that the LLM-provided value will be saved to.

    For example, if the value of this is "input", and the LLM prompt completion is:

    ```
    Decide on your next course of action:

    Action Type: "Search"
    Action Input: "Olivia Wilde boyfriend"
    ```

    Then the final dict returned will store "Olivia Wilde boyfriend" inside the key
    "input":

    ```
    {
        ...
        "input": "Olivia Wilde boyfriend"
        ...
    }
    ```
    """
    stop: str = '"'
    """The character to stop at when completing the value for this variable.

    This defaults to a single double-quote, so that if your prompt goes

    Action Input: "

    then the LLM can know to terminate its input with a double-quote. If you expect the
    output to be multiline, then you may instead want to go with ''' or ```, and adjust
    the display string accordingly

    This exists here instead of MultipleOutputsPrompter so that each variable can have
    its own custom stop if needed.
    """
    display_suffix: Optional[str] = None
    """The string to put at the end of the display string.

    For example, if you set this to ": \"", and your display is "Action Input", then
    the whole thing will come out as "Action Input: \"" for the LLM to complete

    This exists to allow easy modification of the final display string by simply
    copying over the stopper, without polluting the display string with the stopper.
    """

    @property
    def prompt(self) -> str:
        """Prompt for the LLM to fill in this specific variable."""
        if self.display_suffix:
            return self.display + self.display_suffix
        return self.display

    @property
    def prompt_with_value(self) -> str:
        """Previous prompt + filled-in value for this variable.

        This is a templating string, so it won't have the actual value of the variable,
        just a marker for the template to eventually fill it in with.
        """
        return self.prompt + "{" + self.output_key + "}" + self.stop

    @classmethod
    def for_code(
        cls: Type[VC],
        output_key: str,
        display: str,
        language: Optional[str] = None,
    ) -> VC:
        """Configure a language-specific code prompt."""
        language_str = "" if language is None else language
        display_suffix = f": ```{language_str}\n"
        return cls(
            output_key=output_key,
            display=display,
            display_suffix=display_suffix,
            stop="```",
        )
