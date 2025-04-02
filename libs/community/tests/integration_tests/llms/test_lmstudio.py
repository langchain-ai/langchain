import pytest
from langchain_core.outputs import Generation

from langchain_community.llms.lmstudio import LMStudioClientError, LMStudio

# --- Configuration ---
# **IMPORTANT**: Replace with the actual model identifier from LM Studio
# Example: "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
# Ensure this model is loaded and served by your running LM Studio instance.
LM_STUDIO_MODEL_IDENTIFIER = "phi-3-mini-4k-instruct"  # CHANGE THIS IF NEEDED

# Default LM Studio API endpoint
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"


# --- Fixtures ---

@pytest.fixture(scope="module")
def llm_instance():
    """
    Provides an initialized LMStudio instance for the tests.
    Scope is 'module' to initialize only once per test file run.
    """
    print(f"\n--- Initializing LMStudio Fixture ---")
    print(f"Model: {LM_STUDIO_MODEL_IDENTIFIER}")
    print(f"Endpoint: {LM_STUDIO_BASE_URL}")
    print("Ensure LM Studio server is running with this model loaded.")
    try:
        llm = LMStudio(
            model=LM_STUDIO_MODEL_IDENTIFIER,
            base_url=LM_STUDIO_BASE_URL,
            temperature=0.7,
            max_tokens=150,  # Default max_tokens for fixture
        )
        # Optional: Add a quick check here to see if the server is reachable
        # For example, a very short generation or a specific health check endpoint if available.
        # This makes fixture setup fail faster if the server isn't ready.
        # llm.invoke("Hi", max_tokens=5) # Example quick check
        return llm
    except Exception as e:
        # If fixture setup fails, fail the whole test module session
        pytest.fail(
            f"Failed to initialize LMStudio fixture: {e}. Is LM Studio running with the correct model at {LM_STUDIO_BASE_URL}?",
            pytrace=True)


# --- Test Functions ---

# Optional: A basic reachability test
def test_server_reachability(llm_instance):
    """Tests basic connectivity by making a very short request."""
    print("\n--- Test: Server Reachability ---")
    try:
        # Use a short, simple prompt and low max_tokens
        response = llm_instance.invoke("Say 'ok'", max_tokens=10, temperature=0.1)
        print(f"Reachability response: {response}")
        assert isinstance(response, str)
        # We don't strictly check content ("ok") as models can vary,
        # just that we got *some* string response without error.
        assert len(response.strip()) > 0
    except LMStudioClientError as e:
        pytest.fail(f"LMStudio server ({LM_STUDIO_BASE_URL}) unreachable or model not loaded correctly: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected error during reachability test: {e}")


def test_sync_generate(llm_instance):
    """Tests synchronous generation using invoke."""
    print("\n--- Test: Generate (Sync) ---")
    prompt = "What are the main components of a modern computer?"
    print(f"Prompt: {prompt}")
    try:
        response = llm_instance.invoke(prompt, temperature=0.5, max_tokens=100)  # Override fixture max_tokens if needed
        print(f"Response: {response}")
        assert isinstance(response, str)
        assert len(response.strip()) > 10  # Assert a reasonably long response
    except LMStudioClientError as e:
        pytest.fail(f"LMStudioClientError during sync generate: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected error during sync generate: {e}")


def test_sync_stream(llm_instance):
    """Tests synchronous streaming."""
    print("\n--- Test: Stream (Sync) ---")
    prompt = "Write a short story about a curious cat exploring a garden."
    print(f"Prompt: {prompt}")
    print("Response Stream:")
    full_streamed_response = ""
    chunks_received = 0
    try:
        # Use a stop sequence and specific max_tokens
        for chunk in llm_instance.stream(prompt, stop=["."], max_tokens=80):
            print(chunk, end="", flush=True)  # Print for visual feedback
            assert isinstance(chunk, str)  # Each chunk should be a string
            full_streamed_response += chunk
            chunks_received += 1
        print("\n(End of Sync Stream)")

        assert chunks_received > 0, "Stream did not produce any chunks."
        assert len(full_streamed_response.strip()) > 10, "Streamed response is too short or empty."
        # Optional: Check if stop sequence worked (response might not end *exactly* with '.')
        # assert not full_streamed_response.endswith('.') or full_streamed_response.strip() == ''

    except LMStudioClientError as e:
        pytest.fail(f"LMStudioClientError during sync stream: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected error during sync stream: {e}")


@pytest.mark.asyncio
async def test_async_generate(llm_instance):
    """Tests asynchronous generation using agenerate."""
    print("\n--- Test: Generate (Async) ---")
    prompts_async = [
        "List two advantages of renewable energy.",
        "What is the capital of Japan?"
    ]
    print(f"Prompts: {prompts_async}")
    try:
        results = await llm_instance.agenerate(prompts_async, max_tokens=50)  # Specific max_tokens for this test
        print(f"Async Generate Results: {results}")

        assert results is not None
        assert hasattr(results, 'generations')
        assert len(results.generations) == len(prompts_async), "Incorrect number of generation lists returned."

        for i, gen_list in enumerate(results.generations):
            assert isinstance(gen_list, list)
            assert len(gen_list) >= 1, f"No generation found for prompt {i + 1}"
            generation = gen_list[0]  # Get the first generation result
            assert hasattr(generation, 'text')
            assert isinstance(generation, Generation)
            assert len(generation.text) > 0, f"Empty response for prompt {i + 1}"
            print(f"Response {i + 1}: {generation.text}")
            assert generation.generation_info is not None

        # Optional: Check llm_output structure
        assert hasattr(results, 'llm_output')


    except LMStudioClientError as e:
        pytest.fail(f"LMStudioClientError during async generate: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected error during async generate: {e}")


@pytest.mark.asyncio
async def test_async_stream(llm_instance):
    """Tests asynchronous streaming using astream."""
    print("\n--- Test: Stream (Async) ---")
    prompt = "Explain the water cycle simply."
    print(f"Prompt: {prompt}")
    print("Response Stream:")
    full_async_streamed_response = ""
    chunks_received = 0
    try:
        # Use a specific temperature for this test
        async for chunk in llm_instance.astream(prompt, temperature=0.9, max_tokens=70):
            print(chunk, end="", flush=True)  # Print for visual feedback
            assert isinstance(chunk, str)
            full_async_streamed_response += chunk
            chunks_received += 1
        print("\n(End of Async Stream)")

        assert chunks_received > 0, "Async stream did not produce any chunks."
        assert len(full_async_streamed_response.strip()) > 10, "Async streamed response is too short or empty."

    except LMStudioClientError as e:
        pytest.fail(f"LMStudioClientError during async stream: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected error during async stream: {e}")
