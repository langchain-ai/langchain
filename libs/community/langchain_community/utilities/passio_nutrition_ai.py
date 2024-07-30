"""Util that invokes the Passio Nutrition AI API."""

from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional, final

import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.utils import get_from_dict_or_env


class NoDiskStorage:
    """Mixin to prevent storing on disk."""

    @final
    def __getstate__(self) -> None:
        raise AttributeError("Do not store on disk.")

    @final
    def __setstate__(self, state: Any) -> None:
        raise AttributeError("Do not store on disk.")


try:
    from tenacity import (
        retry,
        retry_if_result,
        stop_after_attempt,
        wait_exponential,
        wait_random,
    )
except ImportError:
    # No retries if tenacity is not installed.
    def retry_fallback(
        f: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Callable[..., Any]:
        return f

    def stop_after_attempt_fallback(n: int) -> None:
        return None

    def wait_random_fallback(a: float, b: float) -> None:
        return None

    def wait_exponential_fallback(
        multiplier: float = 1, min: float = 0, max: float = float("inf")
    ) -> None:
        return None


def is_http_retryable(rsp: requests.Response) -> bool:
    """Check if a HTTP response is retryable."""
    return bool(rsp) and rsp.status_code in [408, 425, 429, 500, 502, 503, 504]


class ManagedPassioLifeAuth(NoDiskStorage):
    """Manage the token for the NutritionAI API."""

    _access_token_expiry: Optional[datetime]

    def __init__(self, subscription_key: str):
        self.subscription_key = subscription_key
        self._last_token = None
        self._access_token_expiry = None
        self._access_token = None
        self._customer_id = None

    @property
    def headers(self) -> dict:
        if not self.is_valid_now():
            self.refresh_access_token()
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Passio-ID": self._customer_id,
        }

    def is_valid_now(self) -> bool:
        return (
            self._access_token is not None
            and self._customer_id is not None
            and self._access_token_expiry is not None
            and self._access_token_expiry > datetime.now()
        )

    @retry(
        retry=retry_if_result(is_http_retryable),
        stop=stop_after_attempt(4),
        wait=wait_random(0, 0.3) + wait_exponential(multiplier=1, min=0.1, max=2),
    )
    def _http_get(self, subscription_key: str) -> requests.Response:
        return requests.get(
            f"https://api.passiolife.com/v2/token-cache/napi/oauth/token/{subscription_key}"
        )

    def refresh_access_token(self) -> None:
        """Refresh the access token for the NutritionAI API."""
        rsp = self._http_get(self.subscription_key)
        if not rsp:
            raise ValueError("Could not get access token")
        self._last_token = token = rsp.json()
        self._customer_id = token["customer_id"]
        self._access_token = token["access_token"]
        self._access_token_expiry = (
            datetime.now()
            + timedelta(seconds=token["expires_in"])
            - timedelta(seconds=5)
        )
        # 5 seconds: approximate time for a token refresh to be processed.


DEFAULT_NUTRITIONAI_API_URL = (
    "https://api.passiolife.com/v2/products/napi/food/search/advanced"
)


class NutritionAIAPI(BaseModel):
    """Wrapper for the Passio Nutrition AI API."""

    nutritionai_subscription_key: str
    nutritionai_api_url: str = Field(default=DEFAULT_NUTRITIONAI_API_URL)
    more_kwargs: dict = Field(default_factory=dict)
    auth_: ManagedPassioLifeAuth

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @retry(
        retry=retry_if_result(is_http_retryable),
        stop=stop_after_attempt(4),
        wait=wait_random(0, 0.3) + wait_exponential(multiplier=1, min=0.1, max=2),
    )
    def _http_get(self, params: dict) -> requests.Response:
        return requests.get(
            self.nutritionai_api_url,
            headers=self.auth_.headers,
            params=params,  # type: ignore
        )

    def _api_call_results(self, search_term: str) -> dict:
        """Call the NutritionAI API and return the results."""
        rsp = self._http_get({"term": search_term, **self.more_kwargs})
        if not rsp:
            raise ValueError("Could not get NutritionAI API results")
        rsp.raise_for_status()
        return rsp.json()

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and endpoint exists in environment."""
        nutritionai_subscription_key = get_from_dict_or_env(
            values, "nutritionai_subscription_key", "NUTRITIONAI_SUBSCRIPTION_KEY"
        )
        values["nutritionai_subscription_key"] = nutritionai_subscription_key

        nutritionai_api_url = get_from_dict_or_env(
            values,
            "nutritionai_api_url",
            "NUTRITIONAI_API_URL",
            DEFAULT_NUTRITIONAI_API_URL,
        )
        values["nutritionai_api_url"] = nutritionai_api_url

        values["auth_"] = ManagedPassioLifeAuth(nutritionai_subscription_key)
        return values

    def run(self, query: str) -> Optional[Dict]:
        """Run query through NutrtitionAI API and parse result."""
        results = self._api_call_results(query)
        if results and len(results) < 1:
            return None
        return results
