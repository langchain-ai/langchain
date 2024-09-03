import json
import os
from typing import Dict, Optional


class Portkey:
    """Portkey configuration.

    Attributes:
        base: The base URL for the Portkey API.
          Default: "https://api.portkey.ai/v1/proxy"
    """

    base: str = "https://api.portkey.ai/v1/proxy"

    @staticmethod
    def Config(
        api_key: str,
        trace_id: Optional[str] = None,
        environment: Optional[str] = None,
        user: Optional[str] = None,
        organisation: Optional[str] = None,
        prompt: Optional[str] = None,
        retry_count: Optional[int] = None,
        cache: Optional[str] = None,
        cache_force_refresh: Optional[str] = None,
        cache_age: Optional[int] = None,
    ) -> Dict[str, str]:
        assert retry_count is None or retry_count in range(
            1, 6
        ), "retry_count must be an integer and in range [1, 2, 3, 4, 5]"
        assert cache is None or cache in [
            "simple",
            "semantic",
        ], "cache must be 'simple' or 'semantic'"
        assert cache_force_refresh is None or (
            isinstance(cache_force_refresh, str)
            and cache_force_refresh in ["True", "False"]
        ), "cache_force_refresh must be 'True' or 'False'"
        assert cache_age is None or isinstance(
            cache_age, int
        ), "cache_age must be an integer"

        os.environ["OPENAI_API_BASE"] = Portkey.base

        headers = {
            "x-portkey-api-key": api_key,
            "x-portkey-mode": "proxy openai",
        }

        if trace_id:
            headers["x-portkey-trace-id"] = trace_id
        if retry_count:
            headers["x-portkey-retry-count"] = str(retry_count)
        if cache:
            headers["x-portkey-cache"] = cache
        if cache_force_refresh:
            headers["x-portkey-cache-force-refresh"] = cache_force_refresh
        if cache_age:
            headers["Cache-Control"] = f"max-age:{str(cache_age)}"

        metadata = {}
        if environment:
            metadata["_environment"] = environment
        if user:
            metadata["_user"] = user
        if organisation:
            metadata["_organisation"] = organisation
        if prompt:
            metadata["_prompt"] = prompt

        if metadata:
            headers.update({"x-portkey-metadata": json.dumps(metadata)})

        return headers
