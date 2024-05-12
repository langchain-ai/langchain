import requests
from langchain_core.outputs.generation import GenerationChunk
from langchain_core.outputs.llm_result import LLMResult
from pytest import MonkeyPatch

from langchain_community.llms.ollama import Ollama


def mock_response_stream():  # type: ignore[no-untyped-def]
    mock_response = [b'{ "response": "Response chunk 1" }']

    class MockRaw:
        def read(self, chunk_size):  # type: ignore[no-untyped-def]
            try:
                return mock_response.pop()
            except IndexError:
                return None

    response = requests.Response()
    response.status_code = 200
    response.raw = MockRaw()
    return response


def test_pass_headers_if_provided(monkeypatch: MonkeyPatch) -> None:
    llm = Ollama(
        base_url="https://ollama-hostname:8000",
        model="foo",
        headers={
            "Authorization": "Bearer TEST-TOKEN-VALUE",
            "Referer": "https://application-host",
        },
        timeout=300,
    )

    def mock_post(url, headers, json, stream, timeout):  # type: ignore[no-untyped-def]
        assert url == "https://ollama-hostname:8000/api/generate"
        assert headers == {
            "Content-Type": "application/json",
            "Authorization": "Bearer TEST-TOKEN-VALUE",
            "Referer": "https://application-host",
        }
        assert json is not None
        assert stream is True
        assert timeout == 300

        return mock_response_stream()

    monkeypatch.setattr(requests, "post", mock_post)

    llm.invoke("Test prompt")


def test_handle_if_headers_not_provided(monkeypatch: MonkeyPatch) -> None:
    llm = Ollama(base_url="https://ollama-hostname:8000", model="foo", timeout=300)

    def mock_post(url, headers, json, stream, timeout):  # type: ignore[no-untyped-def]
        assert url == "https://ollama-hostname:8000/api/generate"
        assert headers == {
            "Content-Type": "application/json",
        }
        assert json is not None
        assert stream is True
        assert timeout == 300

        return mock_response_stream()

    monkeypatch.setattr(requests, "post", mock_post)

    llm.invoke("Test prompt")


def test_handle_kwargs_top_level_parameters(monkeypatch: MonkeyPatch) -> None:
    """Test that top level params are sent to the endpoint as top level params"""
    llm = Ollama(base_url="https://ollama-hostname:8000", model="foo", timeout=300)

    def mock_post(url, headers, json, stream, timeout):  # type: ignore[no-untyped-def]
        assert url == "https://ollama-hostname:8000/api/generate"
        assert headers == {
            "Content-Type": "application/json",
        }
        assert json == {
            "format": None,
            "images": None,
            "model": "test-model",
            "options": {
                "mirostat": None,
                "mirostat_eta": None,
                "mirostat_tau": None,
                "num_ctx": None,
                "num_gpu": None,
                "num_thread": None,
                "num_predict": None,
                "repeat_last_n": None,
                "repeat_penalty": None,
                "stop": None,
                "temperature": None,
                "tfs_z": None,
                "top_k": None,
                "top_p": None,
            },
            "prompt": "Test prompt",
            "system": "Test system prompt",
            "template": None,
            "keep_alive": None,
            "preload": False,
        }
        assert stream is True
        assert timeout == 300

        return mock_response_stream()

    monkeypatch.setattr(requests, "post", mock_post)

    llm.invoke("Test prompt", model="test-model", system="Test system prompt")


def test_handle_kwargs_with_unknown_param(monkeypatch: MonkeyPatch) -> None:
    """
    Test that params that are not top level params will be sent to the endpoint
    as options
    """
    llm = Ollama(base_url="https://ollama-hostname:8000", model="foo", timeout=300)

    def mock_post(url, headers, json, stream, timeout):  # type: ignore[no-untyped-def]
        assert url == "https://ollama-hostname:8000/api/generate"
        assert headers == {
            "Content-Type": "application/json",
        }
        assert json == {
            "format": None,
            "images": None,
            "model": "foo",
            "options": {
                "mirostat": None,
                "mirostat_eta": None,
                "mirostat_tau": None,
                "num_ctx": None,
                "num_gpu": None,
                "num_thread": None,
                "num_predict": None,
                "repeat_last_n": None,
                "repeat_penalty": None,
                "stop": None,
                "temperature": 0.8,
                "tfs_z": None,
                "top_k": None,
                "top_p": None,
                "unknown": "Unknown parameter value",
            },
            "prompt": "Test prompt",
            "system": None,
            "template": None,
            "keep_alive": None,
            "preload": False,
        }
        assert stream is True
        assert timeout == 300

        return mock_response_stream()

    monkeypatch.setattr(requests, "post", mock_post)

    llm.invoke("Test prompt", unknown="Unknown parameter value", temperature=0.8)


def test_handle_kwargs_with_options(monkeypatch: MonkeyPatch) -> None:
    """
    Test that if options provided it will be sent to the endpoint as options,
    ignoring other params that are not top level params.
    """
    llm = Ollama(base_url="https://ollama-hostname:8000", model="foo", timeout=300)

    def mock_post(url, headers, json, stream, timeout):  # type: ignore[no-untyped-def]
        assert url == "https://ollama-hostname:8000/api/generate"
        assert headers == {
            "Content-Type": "application/json",
        }
        assert json == {
            "format": None,
            "images": None,
            "model": "test-another-model",
            "options": {"unknown_option": "Unknown option value"},
            "prompt": "Test prompt",
            "system": None,
            "template": None,
            "keep_alive": None,
            "preload": False,
        }
        assert stream is True
        assert timeout == 300

        return mock_response_stream()

    monkeypatch.setattr(requests, "post", mock_post)

    llm.invoke(
        "Test prompt",
        model="test-another-model",
        options={"unknown_option": "Unknown option value"},
        unknown="Unknown parameter value",
        temperature=0.8,
    )


def test_preload_true_initializes_model(monkeypatch: MonkeyPatch) -> None:
    # Create a mock function to replace the preload_model method
    def mock_preload():  # type: ignore[no-untyped-def]
        pass

    monkeypatch.setattr(
        "langchain_community.llms.ollama.Ollama.preload_model", mock_preload
    )

    # Use a counter to track calls to the mock function
    call_count = 0

    def count_calls(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1

    monkeypatch.setattr(
        "langchain_community.llms.ollama.Ollama.preload_model", count_calls
    )
    Ollama(model="llama3", preload=True)
    assert call_count == 1, "`preload_model()` should be called once when preload=True"


def test_preload_false_does_not_initialize_model(monkeypatch: MonkeyPatch) -> None:
    # Create a mock function to replace the preload_model method
    def mock_preload():  # type: ignore[no-untyped-def]
        pass

    monkeypatch.setattr(
        "langchain_community.llms.ollama.Ollama.preload_model", mock_preload
    )

    # Use a counter to track calls to the mock function
    call_count = 0

    def count_calls(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1

    monkeypatch.setattr(
        "langchain_community.llms.ollama.Ollama.preload_model", count_calls
    )
    Ollama(model="llama3", preload=False)
    assert call_count == 0


def test_preload_not_set_does_not_initialize_model(monkeypatch: MonkeyPatch) -> None:
    # Create a mock function to replace the preload_model method
    def mock_preload():  # type: ignore[no-untyped-def]
        pass

    monkeypatch.setattr(
        "langchain_community.llms.ollama.Ollama.preload_model", mock_preload
    )

    # Use a counter to track calls to the mock function
    call_count = 0

    def count_calls(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1

    monkeypatch.setattr(
        "langchain_community.llms.ollama.Ollama.preload_model", count_calls
    )
    Ollama(model="llama3")
    assert call_count == 0


def test_preload_true_effectiveness(monkeypatch: MonkeyPatch) -> None:
    """
    Test that initializing Ollama with `preload=True` actually preloads the model,
    potentially reducing the time for the first invocation.
    """

    # Mock the actual API call to simulate model loading and invocation
    def mock_generate(*args, **kwargs) -> LLMResult:  # type: ignore[no-untyped-def]
        # Simulate a delay that would be seen in model loading
        import time

        time.sleep(0.1 if "preload" in kwargs and kwargs["preload"] else 0.05)
        return LLMResult(
            generations=[[GenerationChunk(text="Model loaded and response generated")]]
        )

    monkeypatch.setattr(
        "langchain_community.llms.ollama.Ollama._generate", mock_generate
    )

    # Measure time with `preload=True`
    import time

    start_time = time.time()
    ollama_preload = Ollama(model="llama3", preload=True)
    ollama_preload.invoke("Test prompt")
    elapsed_time_preload = time.time() - start_time

    # Reset and measure time with `preload=False`
    start_time = time.time()
    ollama_no_preload = Ollama(model="llama3", preload=False)
    ollama_no_preload.invoke("Test prompt")
    elapsed_time_no_preload = time.time() - start_time

    assert elapsed_time_preload > elapsed_time_no_preload

    # Reset and measure time without preload
    start_time = time.time()
    ollama_no_preload_param = Ollama(model="llama3")
    ollama_no_preload_param.invoke("Test prompt")
    elapsed_time_no_preload_param = time.time() - start_time

    assert elapsed_time_preload > elapsed_time_no_preload_param


def test_preload_false_effectiveness(monkeypatch: MonkeyPatch) -> None:
    """
    Test that initializing Ollama with `preload=False` does not preload the model,
    and the first invocation takes longer due to model loading.
    """

    # Similar setup as the previous test but reversed logic for assertions
    def mock_generate(*args, **kwargs) -> LLMResult:  # type: ignore[no-untyped-def]
        import time

        time.sleep(0.1)
        return LLMResult(
            generations=[[GenerationChunk(text="Model loaded and response generated")]]
        )

    monkeypatch.setattr(
        "langchain_community.llms.ollama.Ollama._generate", mock_generate
    )

    # Only measure time without preload as this test focuses on the negative case
    import time

    start_time = time.time()
    ollama_no_preload = Ollama(model="llama3", preload=False)
    ollama_no_preload.invoke("Test prompt")
    elapsed_time_no_preload = time.time() - start_time

    assert elapsed_time_no_preload > 0
