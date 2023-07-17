"""Adapted from https://github.com/danishprakash/pip-name/blob/master/pip-name"""
from typing import List, Optional

import requests

BASE_URL = "https://pypi.org/pypi"

LENGTH_SUGGESTION = "Name too long (>12)"
UPPERCASE_SUGGESTION = "Use of uppercase letters is discouraged"
SEPARATOR_SUGGESTION = "Use of `-` is discouraged, consider using `_`"
NUMERIC_SUGGESTION = "Use of numbers is discouraged"


def lint_name(name: str) -> List[str]:
    """Check name against PEP8's naming conventions"""
    suggestions = []

    if len(name) > 12:
        suggestions.append(LENGTH_SUGGESTION)
    if "-" in name or " " in name:
        suggestions.append(SEPARATOR_SUGGESTION)
    if any(x.isupper() for x in name):
        suggestions.append(UPPERCASE_SUGGESTION)
    if any(x.isnumeric() for x in name):
        suggestions.append(NUMERIC_SUGGESTION)

    return suggestions


def is_name_taken(name: str) -> bool:
    """Check module filename for conflict"""
    response = request_pypi(name)

    if response:
        module_name = response.get("info").get("package_url").split("/")[-2]  # type: ignore  # noqa: E501
        return name.lower() == module_name.lower()

    return False


def request_pypi(name: str) -> Optional[dict]:
    """Request response from PyPi API."""
    target_url = f"{BASE_URL}/{name}/json"
    response = requests.get(target_url)

    return response.json() if response.status_code != 404 else None
