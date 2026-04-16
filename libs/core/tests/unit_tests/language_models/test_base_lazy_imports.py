"""Test that heavy optional dependencies are not imported at module level."""

import sys


def test_transformers_not_imported_on_base_import() -> None:
    """Verify that importing BaseChatModel does not trigger a transformers import.

    Regression test for https://github.com/langchain-ai/langchain/issues/36835.

    When ``transformers`` is installed, the previous top-level ``try/except``
    import added 300-500 ms to every ``from langchain_core.language_models
    import BaseChatModel`` call.  Moving the import inside ``get_tokenizer()``
    avoids the cost until the tokenizer is actually needed.
    """
    # Remove any cached transformers import so we can detect a fresh import.
    had_transformers = "transformers" in sys.modules
    if had_transformers:
        saved = sys.modules.pop("transformers")

    try:
        # Re-import the module to simulate a fresh load.
        import importlib

        import langchain_core.language_models.base as base_mod

        importlib.reload(base_mod)

        # transformers should NOT have been imported as a side-effect.
        assert "transformers" not in sys.modules, (
            "transformers was imported at module level; "
            "it should only be imported lazily inside get_tokenizer()"
        )
    finally:
        # Restore original state.
        if had_transformers:
            sys.modules["transformers"] = saved
