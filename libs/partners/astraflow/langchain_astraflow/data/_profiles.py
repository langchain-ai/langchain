"""Model profiles for Astraflow.

Astraflow aggregates 200+ models from multiple providers.
This file is intentionally sparse — model profiles are fetched from the
Astraflow API at runtime or can be populated via the langchain-profiles CLI.
"""

from typing import Any

_PROFILES: dict[str, dict[str, Any]] = {}
