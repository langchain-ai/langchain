"""Util that calls AskNews api."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env


class AskNewsAPIWrapper(BaseModel):
    """Wrapper for AskNews API."""

    asknews_sync: Any  #: :meta private:
    asknews_async: Any  #: :meta private:
    asknews_client_id: Optional[str] = None
    """Client ID for the AskNews API."""
    asknews_client_secret: Optional[str] = None
    """Client Secret for the AskNews API."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api credentials and python package exists in environment."""

        asknews_client_id = get_from_dict_or_env(
            values, "asknews_client_id", "ASKNEWS_CLIENT_ID"
        )
        asknews_client_secret = get_from_dict_or_env(
            values, "asknews_client_secret", "ASKNEWS_CLIENT_SECRET"
        )

        try:
            import asknews_sdk

        except ImportError:
            raise ImportError(
                "AskNews python package not found. "
                "Please install it with `pip install asknews`."
            )

        an_sync = asknews_sdk.AskNewsSDK(
            client_id=asknews_client_id,
            client_secret=asknews_client_secret,
            scopes=["news"],
        )
        an_async = asknews_sdk.AsyncAskNewsSDK(
            client_id=asknews_client_id,
            client_secret=asknews_client_secret,
            scopes=["news"],
        )

        values["asknews_sync"] = an_sync
        values["asknews_async"] = an_async
        values["asknews_client_id"] = asknews_client_id
        values["asknews_client_secret"] = asknews_client_secret

        return values

    def search_news(
        self, query: str, max_results: int = 10, hours_back: int = 0
    ) -> str:
        """Search news in AskNews API synchronously."""
        if hours_back > 48:
            method = "kw"
            historical = True
            start = int((datetime.now() - timedelta(hours=hours_back)).timestamp())
            stop = int(datetime.now().timestamp())
        else:
            historical = False
            method = "nl"
            start = None
            stop = None

        response = self.asknews_sync.news.search_news(
            query=query,
            n_articles=max_results,
            method=method,
            historical=historical,
            start_timestamp=start,
            end_timestamp=stop,
            return_type="string",
        )
        return response.as_string

    async def asearch_news(
        self, query: str, max_results: int = 10, hours_back: int = 0
    ) -> str:
        """Search news in AskNews API asynchronously."""
        if hours_back > 48:
            method = "kw"
            historical = True
            start = int((datetime.now() - timedelta(hours=hours_back)).timestamp())
            stop = int(datetime.now().timestamp())
        else:
            historical = False
            method = "nl"
            start = None
            stop = None

        response = await self.asknews_async.news.search_news(
            query=query,
            n_articles=max_results,
            method=method,
            historical=historical,
            start_timestamp=start,
            end_timestamp=stop,
            return_type="string",
        )
        return response.as_string
