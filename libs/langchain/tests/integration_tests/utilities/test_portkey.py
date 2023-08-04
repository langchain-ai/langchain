import json

from langchain.utilities import Portkey


def test_Config() -> None:
    headers = Portkey.Config(
        api_key="test_api_key",
        environment="test_environment",
        user="test_user",
        organisation="test_organisation",
        prompt="test_prompt",
        retry_count=3,
        trace_id="test_trace_id",
        cache="simple",
        cache_force_refresh="True",
        cache_age=3600,
    )

    assert headers["x-portkey-api-key"] == "test_api_key"
    assert headers["x-portkey-trace-id"] == "test_trace_id"
    assert headers["x-portkey-retry-count"] == "3"
    assert headers["x-portkey-cache"] == "simple"
    assert headers["x-portkey-cache-force-refresh"] == "True"
    assert headers["Cache-Control"] == "max-age:3600"

    metadata = json.loads(headers["x-portkey-metadata"])
    assert metadata["_environment"] == "test_environment"
    assert metadata["_user"] == "test_user"
    assert metadata["_organisation"] == "test_organisation"
    assert metadata["_prompt"] == "test_prompt"
