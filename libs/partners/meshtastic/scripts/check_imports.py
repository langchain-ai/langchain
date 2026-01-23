"""Check that all imports work correctly."""

import sys

try:
    from langchain_meshtastic import (  # noqa: F401
        MeshtasticSendInput,
        MeshtasticSendTool,
    )
except ImportError as e:
    print(f"Import error: {e}")  # noqa: T201
    sys.exit(1)

print("All imports successful!")  # noqa: T201
