"""Test that all integrations compile."""

import pytest


@pytest.mark.compile
def test_compile_all():
    """Test that all integration modules can be imported successfully."""

    # Test basic imports
    import langchain_llamastack  # noqa: F401
    from langchain_llamastack import LlamaStackSafety  # noqa: F401
    from langchain_llamastack import SafetyResult  # noqa: F401
    from langchain_llamastack import check_llamastack_status  # noqa: F401
    from langchain_llamastack import create_llamastack_llm  # noqa: F401
    from langchain_llamastack import get_llamastack_models  # noqa: F401

    # Test safety and moderation hooks
    from langchain_llamastack.input_output_safety_moderation_hooks import (  # noqa: F401
        SafeLLMWrapper,
        create_input_safety_hook,
        create_input_moderation_hook,
        create_output_safety_hook,
        create_output_moderation_hook,
        create_safe_llm_with_all_hooks,
    )

    assert True  # If we get here without import errors, compilation succeeded
