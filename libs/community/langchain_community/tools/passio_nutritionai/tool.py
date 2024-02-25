"""Tool for the Passio Nutrition AI API."""

from typing import Optional
from pydantic import Field

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool

from langchain_community.utilities.passio_nutritionai import NutritionAIAPI


class NutritionAI(BaseTool):
    """Tool that queries the Passio Nutrition AI API."""

    name: str = "nutritionai_advanced_search"
    description: str = (
        "A wrapper around the Passio Nutrition AI. "
        "Useful to retrieve nutrition facts. "
        "Input should be a search query."
    )
    api_wrapper: NutritionAIAPI

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return self.api_wrapper.run(query)


if __name__ == "__main__":
    # Provide the api_wrapper when creating an instance of NutritionAI
    tool = NutritionAI(api_wrapper=NutritionAIAPI())

    # Run the tool
    print(tool.run("chicken tikka masala"))