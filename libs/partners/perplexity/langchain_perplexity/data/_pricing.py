"""Perplexity API pricing data.

Pricing is per 1 million tokens unless otherwise specified.
Source: https://docs.perplexity.ai/guides/pricing
"""

from typing import TypedDict


class ModelPricing(TypedDict):
    """Pricing structure for a Perplexity model."""

    input_cost_per_million: float
    output_cost_per_million: float
    citation_cost_per_million: float | None
    reasoning_cost_per_million: float | None
    search_cost_per_thousand: float | None


# Pricing data as of December 2024
PERPLEXITY_PRICING: dict[str, ModelPricing] = {
    "sonar": {
        "input_cost_per_million": 1.0,
        "output_cost_per_million": 1.0,
        "citation_cost_per_million": None,
        "reasoning_cost_per_million": None,
        "search_cost_per_thousand": None,
    },
    "sonar-pro": {
        "input_cost_per_million": 3.0,
        "output_cost_per_million": 15.0,
        "citation_cost_per_million": None,
        "reasoning_cost_per_million": None,
        "search_cost_per_thousand": None,
    },
    "sonar-reasoning": {
        "input_cost_per_million": 1.0,
        "output_cost_per_million": 5.0,
        "citation_cost_per_million": None,
        "reasoning_cost_per_million": None,
        "search_cost_per_thousand": None,
    },
    "sonar-reasoning-pro": {
        "input_cost_per_million": 2.0,
        "output_cost_per_million": 8.0,
        "citation_cost_per_million": None,
        "reasoning_cost_per_million": None,
        "search_cost_per_thousand": None,
    },
    "sonar-deep-research": {
        "input_cost_per_million": 2.0,
        "output_cost_per_million": 8.0,
        "citation_cost_per_million": 2.0,
        "reasoning_cost_per_million": 3.0,
        "search_cost_per_thousand": 5.0,
    },
}


def get_model_pricing(model_name: str) -> ModelPricing | None:
    """Get pricing for a specific model.

    Args:
        model_name: The name of the Perplexity model.

    Returns:
        ModelPricing dict if the model is found, None otherwise.
    """
    return PERPLEXITY_PRICING.get(model_name)
