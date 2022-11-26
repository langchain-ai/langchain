"""BasePrompt schema definition."""
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from langchain.formatting import formatter

DEFAULT_FORMATTER_MAPPING = {
    "f-string": formatter.format,
}


def check_valid_template(
    template: str, template_format: str, input_variables: List[str]
) -> None:
    """Check that template string is valid."""
    if template_format not in DEFAULT_FORMATTER_MAPPING:
        valid_formats = list(DEFAULT_FORMATTER_MAPPING)
        raise ValueError(
            f"Invalid template format. Got `{template_format}`;"
            f" should be one of {valid_formats}"
        )
    dummy_inputs = {input_variable: "foo" for input_variable in input_variables}
    try:
        formatter_func = DEFAULT_FORMATTER_MAPPING[template_format]
        formatter_func(template, **dummy_inputs)
    except KeyError:
        raise ValueError("Invalid prompt schema.")


class BasePromptTemplate(ABC):
    """Base prompt should expose the format method, returning a prompt."""

    input_variables: List[str]
    """A list of the names of the variables the prompt template expects."""

    @abstractmethod
    def format(self, **kwargs: Any) -> str:
        """Format the prompt with the inputs.

        Args:
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.

        Example:

        .. code-block:: python

            prompt.format(variable1="foo")
        """

    @abstractmethod
    def _prompt_dict(self, save_path: str) -> Dict:
        """Return a dictionary of the prompt."""

    def save(self, save_path: Union[Path, str], file_name: Optional[str] = None) -> str:
        """Save the prompt.

        Args:
            save_path: Path to directory to save prompt to.
            file_name: Name of file in directory to save prompt to.
                                    Can be of type json or yaml
        Returns:
            The name of the saved prompt file.

        Example:

        .. code-block:: python

            prompt.save(save_path="path/")
        """
        # Convert file to Path object.
        if isinstance(save_path, str):
            file_path = Path(save_path)
        else:
            file_path = save_path
        file_path.mkdir(parents=True, exist_ok=True)

        # Fetch dictionary to save
        prompt_dict = self._prompt_dict(save_path)

        save_name = file_name or "prompt.yaml"
        file_path = file_path / save_name
        if file_path.suffix == ".json":
            with open(file_path, "w") as f:
                f.write(json.dumps(prompt_dict, indent=4))
        elif file_path.suffix == ".yaml":
            with open(file_path, "w") as f:
                yaml.dump(prompt_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"{save_name} must be json or yaml")
        return str(file_path)
