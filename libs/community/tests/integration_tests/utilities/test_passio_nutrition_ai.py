"""Integration test for Bing Search API Wrapper."""
from langchain_community.utilities.passio_nutrition_ai import NutritionAIAPI


def test_call() -> None:
    """Test that call gives the correct answer."""
    search = NutritionAIAPI()
    output = search.run("Chicken tikka masala")
    assert "Chicken tikka masala" in output
