from langchain_core.utils import mustache
from pydantic import BaseModel

from .core import Invoker, Prompty, SimpleModel


class MustacheRenderer(Invoker):
    """Render a mustache template."""

    def __init__(self, prompty: Prompty) -> None:
        self.prompty = prompty

    def invoke(self, data: BaseModel) -> BaseModel:
        if not isinstance(data, SimpleModel):
            raise ValueError("Expected data to be an instance of SimpleModel")
        generated = mustache.render(self.prompty.content, data.item)
        return SimpleModel[str](item=generated)
