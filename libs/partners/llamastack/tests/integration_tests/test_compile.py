"""Test that all integrations compile."""

import pytest


@pytest.mark.compile
def test_compile_all():
    """Test that all integration modules can be imported successfully."""

    # Test basic imports
    import langchain_llamastack  # noqa: F401
    from langchain_llamastack import (  # noqa: F401
        check_llamastack_status,
        create_llamastack_llm,
        get_llamastack_models,
        LlamaStackSafety,
        SafetyResult,
    )

    # Test safety and moderation hooks
    from langchain_llamastack.input_output_safety_moderation_hooks import (  # noqa: F401
        create_input_moderation_hook,
        create_input_safety_hook,
        create_output_moderation_hook,
        create_output_safety_hook,
        create_safe_llm_with_all_hooks,
        SafeLLMWrapper,
    )

    assert True  # If we get here without import errors, compilation succeeded
