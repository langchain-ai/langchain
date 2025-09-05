"""Test that all integrations compile."""

import pytest


@pytest.mark.compile
def test_compile_all():
    """Test that all integration modules can be imported successfully."""

    # Test basic imports - import and verify module is available
    import langchain_llamastack

    # Verify the module was imported successfully by checking it's not None
    assert langchain_llamastack is not None

    # Verify the module has the expected attributes (basic smoke test)
    assert hasattr(langchain_llamastack, '__version__') or\
    hasattr(langchain_llamastack, '__name__')
