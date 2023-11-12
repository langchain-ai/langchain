"""Util that calls Steam-WebAPI."""

from langchain.pydantic_v1 import BaseModel, Extra


class SteamWebAPIWrapper(BaseModel):
    # Steam WebAPI Implementation will go here...

    # can decide later if needed
    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def run(self, prompt: str = "demo") -> str:
        return prompt
