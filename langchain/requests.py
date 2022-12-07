"""Lightweight wrapper around request library."""
from typing import Optional

import requests
from pydantic import BaseModel


class RequestsWrapper(BaseModel):
    """Lightweight wrapper to partial out everything except the url to hit."""

    headers: Optional[dict] = None

    def run(self, url: str) -> str:
        """Hit the URL and return the text."""
        return requests.get(url, headers=self.headers).text
