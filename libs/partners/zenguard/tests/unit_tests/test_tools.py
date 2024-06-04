import os

import pytest

from langchain_zenguard import Detector, ZenGuardTool


@pytest.fixture()
def zenguard_tool():
    api_key = os.environ.get("ZENGUARD_API_KEY")
    assert api_key, "ZENGUARD_API_KEY is not set"
    return ZenGuardTool()

def assert_successful_response_not_detected(response):
    assert response is not None
    assert "error" not in response, f"API returned an error: {response.get('error')}"
    assert response.get("is_detected") is False, f"Prompt was detected: {response}"


def test_prompt_injection(zenguard_tool):
    prompt = "Simple prompt injection test"
    detectors = [Detector.PROMPT_INJECTION]
    response = zenguard_tool.run({"prompts": [prompt], "detectors": detectors})
    assert_successful_response_not_detected(response)


def test_pii(zenguard_tool):
    prompt = "Simple PII test"
    detectors = [Detector.PII]
    response = zenguard_tool.run({"prompts": [prompt], "detectors": detectors})
    assert_successful_response_not_detected(response)


def test_allowed_topics(zenguard_tool):
    prompt = "Simple allowed topics test"
    detectors = [Detector.ALLOWED_TOPICS]
    response = zenguard_tool.run({"prompts": [prompt], "detectors": detectors})
    assert_successful_response_not_detected(response)


def test_banned_topics(zenguard_tool):
    prompt = "Simple banned topics test"
    detectors = [Detector.BANNED_TOPICS]
    response = zenguard_tool.run({"prompts": [prompt], "detectors": detectors})
    assert_successful_response_not_detected(response)


def test_keywords(zenguard_tool):
    prompt = "Simple keywords test"
    detectors = [Detector.KEYWORDS]
    response = zenguard_tool.run({"prompts": [prompt], "detectors": detectors})
    assert_successful_response_not_detected(response)


def test_secrets(zenguard_tool):
    prompt = "Simple secrets test"
    detectors = [Detector.SECRETS]
    response = zenguard_tool.run({"prompts": [prompt], "detectors": detectors})
    assert_successful_response_not_detected(response)


def test_toxicity(zenguard_tool):
    prompt = "Simple toxicity test"
    detectors = [Detector.TOXICITY]
    response = zenguard_tool.run({"prompts": [prompt], "detectors": detectors})
    assert_successful_response_not_detected(response)
