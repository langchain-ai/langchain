from unittest.mock import MagicMock, patch

import pytest


def test_chat_model_enforces_before_generate() -> None:
    """AIIdentityChatOpenAI calls enforce_access before _generate."""
    with patch(
        "langchain_ai_identity.chat_models.enforce_access",
        return_value={"decision": "allow"},
    ) as mock_enforce, patch(
        "langchain_ai_identity.callback.post_audit",
    ), patch(
        "langchain_openai.ChatOpenAI._generate",
        return_value=MagicMock(),
    ):
        from langchain_ai_identity import AIIdentityChatOpenAI

        llm = AIIdentityChatOpenAI(
            agent_id="test-uuid",
            ai_identity_api_key="aid_sk_test",
            openai_api_key="sk-test",
        )
        llm._generate(messages=[])
        mock_enforce.assert_called_once()


def test_chat_model_deny_raises() -> None:
    """AIIdentityChatOpenAI raises PermissionError when gateway denies."""
    with patch(
        "langchain_ai_identity.chat_models.enforce_access",
        side_effect=PermissionError("denied"),
    ), patch(
        "langchain_ai_identity.callback.post_audit",
    ):
        from langchain_ai_identity import AIIdentityChatOpenAI

        llm = AIIdentityChatOpenAI(
            agent_id="test-uuid",
            ai_identity_api_key="aid_sk_test",
            openai_api_key="sk-test",
        )
        with pytest.raises(PermissionError):
            llm._generate(messages=[])
