"""Verify that public symbols are importable from the top-level package."""


def test_gateway_module_importable() -> None:
    """The _gateway module should be importable."""
    from langchain_ai_identity._gateway import enforce_access, post_audit

    assert callable(enforce_access)
    assert callable(post_audit)


def test_callback_classes_importable() -> None:
    """Callback handler classes should be importable."""
    from langchain_ai_identity.callback import (
        AIIdentityAsyncCallbackHandler,
        AIIdentityCallbackHandler,
    )

    assert AIIdentityCallbackHandler is not None
    assert AIIdentityAsyncCallbackHandler is not None


def test_middleware_importable() -> None:
    """Governance middleware should be importable."""
    from langchain_ai_identity.middleware import AIIdentityGovernanceMiddleware

    assert AIIdentityGovernanceMiddleware is not None
