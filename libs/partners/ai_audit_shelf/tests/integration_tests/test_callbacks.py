import pytest

from langchain_ai_audit_shelf.callbacks import AIAuditCallbackHandler


@pytest.mark.compile
def test_imports() -> None:
    """Placeholder test to verify imports compile correctly."""
    handler = AIAuditCallbackHandler()
    assert handler.actor == "langchain-agent"
