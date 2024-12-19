"""Tool for the Passio Nutrition AI API."""

from typing import Dict, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.passio_nutrition_ai import NutritionAIAPI


class NutritionAIInputs(BaseModel):
    """Inputs to the Passio Nutrition AI tool."""

    query: str = Field(
        description="A query to look up using Passio Nutrition AI, usually a few words."
    )


class NutritionAI(BaseTool):  # type: ignore[override, override]
    """Tool that queries the Passio Nutrition AI API."""

    name: str = "nutritionai_advanced_search"
    description: str = (
        "A wrapper around the Passio Nutrition AI. "
        "Useful to retrieve nutrition facts. "
        "Input should be a search query string."
    )
    api_wrapper: NutritionAIAPI
    args_schema: Type[BaseModel] = NutritionAIInputs

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Optional[Dict]:
        """Use the tool."""
        return self.api_wrapper.run(query)
