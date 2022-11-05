from langchain.llms.base import LLM
from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Extra, root_validator


class ManifestWrapper(BaseModel, LLM):

    client: Any  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that python package exists in environment."""
        try:
            from manifest import Manifest

            if not isinstance(values["client"], Manifest):
                raise ValueError
        except ImportError:
            raise ValueError(
                "Could not import manifest python package. "
                "Please it install it with `pip install manifest`."
            )
        return values

    def __call__(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if stop is not None:
            raise NotImplementedError("Need to check how to do this")
        return self.client.run(prompt)