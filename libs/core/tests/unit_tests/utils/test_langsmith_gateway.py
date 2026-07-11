import httpx

from langchain_core.utils.langsmith_gateway import LangSmithGatewayOAuth


def test_gateway_oauth_replaces_provider_auth_headers() -> None:
    """Gateway OAuth must not be forwarded as a provider credential."""
    auth = LangSmithGatewayOAuth("oauth-token")
    request = httpx.Request(
        "POST",
        "https://gateway.smith.langchain.com/anthropic/v1/messages",
        headers={
            "Authorization": "Bearer provider-key",
            "X-Api-Key": "provider-key",
        },
    )

    authorized_request = next(auth.auth_flow(request))

    assert authorized_request.headers["Authorization"] == "Bearer oauth-token"
    assert "X-Api-Key" not in authorized_request.headers
