import pytest
import requests_mock


@pytest.mark.parametrize("endpoint", ("smart", "research"))
@pytest.mark.requires("sseclient")
def test_invoke(
    endpoint: str, requests_mock: requests_mock.Mocker, monkeypatch: pytest.MonkeyPatch
) -> None:
    from langchain_community.llms import You
    from langchain_community.llms.you import RESEARCH_ENDPOINT, SMART_ENDPOINT

    json = {
        "answer": (
            "A solar eclipse occurs when the Moon passes between the Sun and Earth, "
            "casting a shadow on Earth and ..."
        ),
        "search_results": [
            {
                "url": "https://en.wikipedia.org/wiki/Solar_eclipse",
                "name": "Solar eclipse - Wikipedia",
                "snippet": (
                    "A solar eclipse occurs when the Moon passes "
                    "between Earth and the Sun, thereby obscuring the view of the Sun "
                    "from a small part of Earth, totally or partially. "
                ),
            }
        ],
    }
    request_endpoint = SMART_ENDPOINT if endpoint == "smart" else RESEARCH_ENDPOINT
    requests_mock.post(request_endpoint, json=json)

    monkeypatch.setenv("YDC_API_KEY", "...")

    llm = You(endpoint=endpoint)
    output = llm.invoke("What is a solar eclipse?")
    assert output == json["answer"]
