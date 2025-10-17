import pytest
from langchain_community.chat_models.google_gemini import ChatGoogleGemini

def test_handle_unknown_finish_reason():
    """Ensure ChatGoogleGemini gracefully handles unrecognized FinishReason enum values."""

    # Mock Gemini API part object
    class MockPart:
        text = "Mock message"  # The main message text
        inline_data = None     # Simulate no inline data

    # Mock Gemini API content object: must contain .parts and .role
    class MockContent:
        parts = [MockPart()]   # List of message parts
        role = "model"         # Emulates model output (could also be "user")

    # Mock Gemini API candidate object: represents one possible answer
    class MockCandidate:
        finish_reason = 12              # Simulate an unknown finish reason (int, not Enum)
        content = MockContent()         # Candidate content
        safety_ratings = None           # Needed by LangChain parsing logic
        citation_metadata = None        # Sometimes parsed by LangChain, safer to include

    # Mock Gemini API response object: contains a list of candidates and prompt_feedback
    class MockResponse:
        candidates = [MockCandidate()]  # Single candidate with unknown finish_reason
        prompt_feedback = None          # Not relevant for this test

    # Mock Gemini API client: overrides .generate_content to return our mock response
    class MockClient:
        def generate_content(self, *_, **__):
            return MockResponse()

    mock_client = MockClient()
    llm = ChatGoogleGemini(client=mock_client, model="gemini-pro")
    llm._generative_model = mock_client  # Force internal use of mock client for test

    # The test: Try a normal invocation, which should NOT crash even with unknown finish_reason
    result = llm.invoke("Hello!")
    assert result is not None
    # Ensure the generated info string contains the expected unknown code
    assert "UNKNOWN(12)" in str(result), "Model did not handle unknown FinishReason as expected"
