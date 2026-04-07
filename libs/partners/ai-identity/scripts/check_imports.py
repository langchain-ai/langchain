"""Verify that all public exports can be imported."""

import importlib
import sys


def main() -> None:
    """Check that the package imports correctly."""
    mod = importlib.import_module("langchain_ai_identity")
    expected = [
        "AIIdentityCallbackHandler",
        "AIIdentityAsyncCallbackHandler",
        "AIIdentityChatOpenAI",
        "AIIdentityGovernanceMiddleware",
        "AIIdentityToolkit",
        "create_ai_identity_agent",
    ]
    missing = [name for name in expected if not hasattr(mod, name)]
    if missing:
        print(f"FAIL: Missing exports: {missing}", file=sys.stderr)  # noqa: T201
        sys.exit(1)
    print(f"OK: All {len(expected)} exports found.")  # noqa: T201


if __name__ == "__main__":
    main()
