"""GitHub utilities."""

import http.client
import json
from typing import Optional


def list_packages(*, contains: Optional[str] = None) -> list[str]:
    """List all packages in the langchain repository templates directory."""
    conn = http.client.HTTPSConnection("api.github.com")
    try:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "langchain-cli",
        }

        conn.request(
            "GET",
            "/repos/langchain-ai/langchain/contents/templates",
            headers=headers,
        )
        res = conn.getresponse()

        res_str = res.read()

        data = json.loads(res_str)
        package_names = [
            p["name"] for p in data if p["type"] == "dir" and p["name"] != "docs"
        ]
        return (
            [p for p in package_names if contains in p] if contains else package_names
        )
    finally:
        conn.close()
