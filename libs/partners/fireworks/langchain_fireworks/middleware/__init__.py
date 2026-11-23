"""Middleware for Fireworks models."""

from langchain_fireworks.middleware.prompt_caching import (
    FireworksPromptCachingMiddleware,
)

__all__ = ["FireworksPromptCachingMiddleware"]
