
from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

from langchain_community.utilities.google_translate import GoogleTranslateApiWarper


class GTInput(BaseModel):
    """Input for the Google Translate tool."""

    text: str = Field(description="text to translate")

class GoogleTranslateRun(BaseTool):
    """Tool that queries the Google Translate API."""

    name: str = "google_translate"
    description: str = (
        "A wrapper around Google Translate. "
        "Useful for when you need to translate text. "
        "Input should be a text to translate."
    )

    api_wrapper: GoogleTranslateApiWarper = Field(
        default_factory=GoogleTranslateApiWarper
    )
    args_schema: Type[BaseModel] = GTInput

    def _run(
        self,
        text: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return self.api_wrapper.run(text)