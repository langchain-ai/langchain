from langchain_community.llms.lmstudio import LMStudio

# --- Configuration ---
# **IMPORTANT**: Replace with the actual model identifier from LM Studio
# Ensure this model is loaded and served by your running LM Studio instance.
LM_STUDIO_MODEL_IDENTIFIER = "phi-3-mini-4k-instruct"  # CHANGE THIS IF NEEDED

# Default LM Studio API endpoint
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"


def test_llm_studio_call():
    llm = LMStudio(
        model=LM_STUDIO_MODEL_IDENTIFIER,
        base_url=LM_STUDIO_BASE_URL,
        temperature=0.7,
        max_tokens=150,  # Default max_tokens for fixture
    )
    # Optional: Add a quick check here to see if the server is reachable
    # For example, a very short generation or a specific health check endpoint if available.
    # This makes fixture setup fail faster if the server isn't ready.
    answer = llm.invoke("Hi", max_tokens=5)  # Example quick check
    assert isinstance(answer, str)
    assert len(answer.strip()) > 0
