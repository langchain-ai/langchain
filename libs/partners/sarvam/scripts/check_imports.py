"""Check that the package can be imported."""

import sys
from importlib import import_module


def main() -> None:
    """Check that the package can be imported."""
    try:
        import_module("langchain_sarvam")
        import_module("langchain_sarvam.chat_models")
        sys.exit(0)
    except ImportError as e:
        sys.stderr.write(f"✗ Import failed: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
