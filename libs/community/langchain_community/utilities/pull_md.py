"""Utility that calls the Pull.md API for markdown conversion."""

import logging
from typing import Any, Dict

from pydantic import BaseModel
import requests

logger = logging.getLogger(__name__)


class PullMdAPIWrapper(BaseModel):
    """Wrapper around PullMdAPI to convert URLs to Markdown.

    This utility provides a method to convert a given URL to Markdown format using
    the Pull.md service.
    """

    def convert_url_to_markdown(self, url: str) -> str:
        """Converts a URL to Markdown using the Pull.md service.

        Args:
            url: A string representing the URL to be converted.

        Returns:
            A string containing the Markdown version of the URL's content.

        Raises:
            HTTPError: An error from the requests library for bad HTTP responses.
        """
        try:
            from pull_md import pull_markdown  # Assuming pull_md is installed in the environment
            return pull_markdown(url)
        except ImportError:
            raise ImportError("pull_md package is not installed. Install it with `pip install pull-md`")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to Pull.md API failed: {e}")
            raise

    class Config:
        arbitrary_types_allowed = True