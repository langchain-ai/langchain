"""Code helps to check availability of the name of the project on PyPi

Adapted from https://github.com/danishprakash/pip-name/blob/master/pip-name
"""
from typing import List, Optional

import requests

BASE_URL = "https://pypi.org/pypi"

UPPERCASE_SUGGESTION = "Use of uppercase letters is discouraged"
SEPARATOR_SUGGESTION = "Use of `-` is discouraged, consider using `_`"
NUMERIC_SUGGESTION = "Use of numbers is discouraged"


def _request_pypi(name: str) -> Optional[dict]:
    """Request response from PyPi API.

    Args:
        name (str): Name of the project

    Returns:
        Optional[dict]: Response from PyPi API
    """
    target_url = f"{BASE_URL}/{name}/json"
    response = requests.get(target_url)
    return response.json() if response.status_code != 404 else None


# PUBLIC API


def lint_name(name: str) -> List[str]:
    """Check name against PEP8's naming conventions.

    Args:
        name (str): Name of the project

    Returns:
        List[str]: List of suggestions
    """
    suggestions = []

    if "-" in name or " " in name:
        suggestions.append(SEPARATOR_SUGGESTION)
    if any(x.isupper() for x in name):
        suggestions.append(UPPERCASE_SUGGESTION)
    if any(x.isnumeric() for x in name):
        suggestions.append(NUMERIC_SUGGESTION)

    return suggestions


def is_name_taken(name: str) -> bool:
    """Check module filename for conflict.

    Args:
        name (str): Name of the project

    Returns:
        bool: True if name is taken, False otherwise
    """
    response = _request_pypi(name)

    if response:
        package_url = response.get("info").get("package_url")  # type: ignore
        module_name = package_url.split("/")[-2]
        return name.lower() == module_name.lower()

    return False
