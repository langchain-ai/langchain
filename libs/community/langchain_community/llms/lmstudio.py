from __future__ import annotations

import asyncio
import json
import re  # Import regular expressions for parsing SSE data
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,

)

import aiohttp
import requests
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
# from langchain_core.language_models import BaseLanguageModel # BaseLLM inherits from this
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import GenerationChunk, LLMResult, Generation
from pydantic import ConfigDict  # Import Field, SecretStr for params

# Regex to extract the JSON data part from SSE lines
sse_data_pattern = re.compile(r"data:\s*(.*)")


# Custom Exception for LM Studio specific issues
class LMStudioClientError(Exception):
    """Raised for issues specific to interacting with the LM Studio client manually."""


def _parse_sse_line(line: str) -> Optional[Dict[str, Any]]:
    """Parses a single line from an SSE stream."""
    match = sse_data_pattern.match(line)
    if match:
        data_str = match.group(1).strip()
        if data_str == "[DONE]":
            # Handle the OpenAI standard termination signal if LM Studio sends it
            return {"done": True}
        if data_str:
            try:
                return json.loads(data_str)
            except json.JSONDecodeError:
                # Handle potential malformed JSON
                print(f"Warning: Could not decode JSON from SSE line: {data_str}")
                return None
    return None


def _sse_chunk_to_generation_chunk(chunk_data: Dict[str, Any]) -> GenerationChunk:
    """Convert a parsed SSE JSON chunk dictionary to a GenerationChunk."""
    if not chunk_data or not chunk_data.get("choices"):
        return GenerationChunk(text="")

    choice = chunk_data["choices"][0]
    delta = choice.get("delta", {})
    text = delta.get("content", "")

    generation_info = {}
    if choice.get("finish_reason") is not None:
        generation_info["finish_reason"] = choice["finish_reason"]
    if chunk_data.get("usage") is not None:  # Usage might appear in the last chunk
        generation_info["usage"] = chunk_data["usage"]
    # Add other potential fields from the chunk if needed

    return GenerationChunk(text=text or "", generation_info=generation_info or None)


class LMStudio(BaseLLM):
    """LM Studio LLM client using manual requests (requests/aiohttp).

    Interacts with LM Studio's OpenAI-compatible server endpoint without
    using the 'openai' library, adhering to a restricted import set.

    Ensure LM Studio server is running and a model is loaded.
    Server address defaults to "http://localhost:1234/v1".

    Example:
        .. code-block:: python

            from your_module import LMStudio # Assuming you save this code
            llm = LMStudio(model="loaded-model-identifier", temperature=0.7)
            response = llm.invoke("All Hail Britannia")
            print(response)
    """
    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
    )

    base_url: str = "http://localhost:1234/v1"
    """Base URL for the LM Studio API endpoint."""

    # Add '/chat/completions' to base_url or handle it in methods
    @property
    def _endpoint_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    model: str = "local-model"
    """The identifier of the model loaded in LM Studio.
       Check LM Studio UI for the exact identifier."""

    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    top_p: Optional[float] = 1.0
    stop: Optional[List[str]] = None

    timeout: Optional[int] = 120  # Default timeout for requests
    """Timeout for HTTP requests in seconds."""

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        # Adjusted llm_type to reflect new name
        return "lmstudio"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters (used for caching, logging)."""
        # Should only include parameters that identify the model configuration
        return {
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stop": self.stop,
        }

    def _prepare_payload(
            self,
            prompt: str,
            stream: bool,
            stop: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> Dict[str, Any]:
        """Prepare the JSON payload for the API request EXPLICITLY."""
        # Start with core required items
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
        }

        # Explicitly pull known, serializable parameters from self or kwargs,
        # preferring kwargs if provided at call time.
        # These are parameters expected by the OpenAI /chat/completions endpoint.

        # Temperature
        # CORRECTED DEFAULT VALUE HERE: self -> self
        temperature = kwargs.get("temperature", self.temperature)  # <--- FIXED
        if temperature is not None:
            # Basic type check for safety, although Pydantic handles instance vars
            if isinstance(temperature, (float, int)):
                payload["temperature"] = temperature
            else:
                # This warning should no longer appear for temperature unless
                # an invalid type is explicitly passed in kwargs
                print(f"Warning: Invalid type for 'temperature' ignored: {type(temperature)}")

        # Max Tokens (This one was already correct)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        if max_tokens is not None:
            if isinstance(max_tokens, int):
                payload["max_tokens"] = max_tokens
            else:
                print(f"Warning: Invalid type for 'max_tokens' ignored: {type(max_tokens)}")

        # Top P (This one was already correct)
        top_p = kwargs.get("top_p", self.top_p)
        if top_p is not None:
            if isinstance(top_p, (float, int)):
                payload["top_p"] = top_p
            else:
                print(f"Warning: Invalid type for 'top_p' ignored: {type(top_p)}")

        # Stop Sequences (Logic appears correct)
        # Prioritize the 'stop' argument passed directly to the method (_generate, _stream)
        # Fallback to 'stop' potentially passed in via kwargs (e.g., llm.invoke(..., stop=[]))
        # Fallback finally to the instance's default self.stop
        final_stop = stop
        if final_stop is None:
            final_stop = kwargs.get("stop", self.stop)

        if final_stop is not None:
            # Ensure it's a list of strings, or a single string
            if isinstance(final_stop, str):
                payload["stop"] = [final_stop]  # API usually expects a list
            elif isinstance(final_stop, list) and all(isinstance(s, str) for s in final_stop):
                payload["stop"] = final_stop
            else:
                # Avoid adding invalid types to the payload
                print(f"Warning: Invalid 'stop' sequence format ignored: {final_stop}")

        # --- IMPORTANT ---
        # We specifically DO NOT merge arbitrary kwargs here using update().
        # This prevents non-serializable objects (like the LLM instance itself)
        # that might have been passed down the LangChain call stack via kwargs
        # from entering the payload dictionary.

        # Return the explicitly constructed payload
        return payload

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for the request."""
        # LM Studio usually doesn't require auth, but can uncomment if needed
        # key_value = self.api_key.get_secret_value() if isinstance(self.api_key, SecretStr) else self.api_key
        return {
            "Content-Type": "application/json",
            # "Authorization": f"Bearer {key_value}"
        }

    # --- Synchronous Methods ---

    def _generate(
            self,
            prompts: List[str],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> LLMResult:
        """Call out to LM Studio's inference endpoint (non-streaming) using requests."""
        generations = []
        headers = self._get_headers()

        for prompt in prompts:
            # Uses the corrected _prepare_payload
            payload = self._prepare_payload(prompt, stream=False, stop=stop, **kwargs)
            try:
                response = requests.post(
                    self._endpoint_url,
                    headers=headers,
                    json=payload,  # requests handles serialization of the clean payload
                    timeout=self.timeout,
                )
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

                response_data = response.json()

                if not response_data.get("choices"):
                    raise LMStudioClientError(f"Invalid response structure received: {response_data}")

                # Extract text and generation info
                choice = response_data["choices"][0]
                text = choice.get("message", {}).get("content", "")
                gen_info = {
                    "finish_reason": choice.get("finish_reason"),
                    "usage": response_data.get("usage"),
                    "model": response_data.get("model"),  # Model name echoed back
                }
                # Filter out None values from gen_info
                gen_info = {k: v for k, v in gen_info.items() if v is not None}

                generations.append([Generation(text=text or "", generation_info=gen_info)])  # Ensure text is str

            except requests.exceptions.Timeout as e:
                raise LMStudioClientError(f"Request timed out after {self.timeout}s: {e}") from e
            except requests.exceptions.ConnectionError as e:
                raise LMStudioClientError(
                    f"Connection error: Could not connect to {self._endpoint_url}. Is LM Studio server running? Details: {e}") from e
            except requests.exceptions.HTTPError as e:
                # Attempt to get more detail from the response body on error
                error_detail = "Could not retrieve error details."
                try:
                    error_detail = e.response
                except Exception:
                    pass
                raise LMStudioClientError(
                    f"LM Studio server returned error {e.response.status_code}: {error_detail}"
                ) from e
            except requests.exceptions.RequestException as e:
                raise LMStudioClientError(f"An unexpected request error occurred: {e}") from e
            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:  # Added TypeError
                # Capture potential issues if response structure is unexpected
                response_text = "Could not retrieve response text."
                if 'response' in locals() and hasattr(response, 'text'):
                    response_text = response
                raise LMStudioClientError(
                    f"Failed to parse LM Studio response or invalid structure: {e}. Response text: {response_text[:500]}") from e

        return LLMResult(generations=generations)

    def _stream(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Stream responses from LM Studio endpoint using requests."""
        # Uses the corrected _prepare_payload
        payload = self._prepare_payload(prompt, stream=True, stop=stop, **kwargs)
        headers = self._get_headers()
        response_iter = None  # To manage closing

        try:
            response = requests.post(
                self._endpoint_url,
                headers=headers,
                json=payload,  # requests handles serialization
                stream=True,
                timeout=self.timeout,
            )
            response.raise_for_status()  # Check for initial errors like 4xx
            response_iter = response.iter_lines(decode_unicode=True)

            for line in response_iter:
                if line:  # Filter out keep-alive newlines
                    chunk_data = _parse_sse_line(line)
                    if chunk_data:
                        if chunk_data.get("done"):  # Handle [DONE] signal
                            break
                        gen_chunk = _sse_chunk_to_generation_chunk(chunk_data)
                        yield gen_chunk
                        if run_manager:
                            run_manager.on_llm_new_token(gen_chunk, chunk=gen_chunk)

        except requests.exceptions.Timeout as e:
            raise LMStudioClientError(f"Request stream timed out after {self.timeout}s: {e}") from e
        except requests.exceptions.ConnectionError as e:
            raise LMStudioClientError(
                f"Connection error during stream: Could not connect to {self._endpoint_url}. Is LM Studio server running? Details: {e}") from e
        except requests.exceptions.HTTPError as e:
            error_detail = "Could not retrieve error details."
            try:
                # We might not have the full response body easily in streaming error
                error_detail = e.response
            except Exception:
                pass
            raise LMStudioClientError(
                f"LM Studio server returned error {e.response.status_code} during stream initiation: {error_detail}"
            ) from e
        except requests.exceptions.RequestException as e:
            raise LMStudioClientError(f"An unexpected request error occurred during streaming: {e}") from e
        except Exception as e:
            # Catch potential errors during SSE parsing or chunk processing
            import traceback
            traceback.print_exc()  # Print full traceback for debugging
            raise LMStudioClientError(f"An error occurred processing the stream: {e}") from e
        finally:
            # Ensure the response is closed if connection was established
            if 'response' in locals() and response is not None:
                response.close()

    # --- Asynchronous Methods ---

    async def _agenerate(
            self,
            prompts: List[str],
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> LLMResult:
        """Asynchronous call to LM Studio (non-streaming) using aiohttp."""
        generations = []
        headers = self._get_headers()
        # Consider creating the session per-request or managing it externally
        # for better resource handling in long-running apps.
        connector = aiohttp.TCPConnector(limit_per_host=100)  # Basic connector config
        async with aiohttp.ClientSession(connector=connector) as session:
            for prompt in prompts:
                # Uses the corrected _prepare_payload
                payload = self._prepare_payload(prompt, stream=False, stop=stop, **kwargs)
                response = None  # Define response in outer scope for error handling
                try:
                    async with session.post(
                            self._endpoint_url,
                            headers=headers,
                            json=payload,  # aiohttp handles serialization
                            timeout=aiohttp.ClientTimeout(total=self.timeout)  # Use aiohttp timeout object
                    ) as response:
                        response.raise_for_status()  # Raise ClientResponseError for bad status
                        response_data = await response.json()

                        if not response_data.get("choices"):
                            raise LMStudioClientError(f"Invalid response structure received: {response_data}")

                        # Extract text and generation info
                        choice = response_data["choices"][0]
                        text = choice.get("message", {}).get("content", "")
                        gen_info = {
                            "finish_reason": choice.get("finish_reason"),
                            "usage": response_data.get("usage"),
                            "model": response_data.get("model"),
                        }
                        gen_info = {k: v for k, v in gen_info.items() if v is not None}

                        generations.append(
                            [Generation(text=text or "", generation_info=gen_info)])  # Ensure text is str

                except aiohttp.ClientConnectorError as e:
                    raise LMStudioClientError(
                        f"Connection error: Could not connect to {self._endpoint_url}. Is LM Studio server running? Details: {e}") from e
                except aiohttp.ClientResponseError as e:
                    # Ensure response object exists before trying to read text
                    error_detail = "(Could not retrieve error details)"
                    if response is not None:
                        try:
                            error_detail = await response()
                        except Exception:
                            pass
                    raise LMStudioClientError(
                        f"LM Studio server returned error {e.status}: {error_detail}"
                    ) from e
                except asyncio.TimeoutError as e:  # aiohttp raises asyncio.TimeoutError
                    raise LMStudioClientError(f"Request timed out after {self.timeout}s: {e}") from e
                except aiohttp.ClientError as e:  # Catch other aiohttp client errors
                    raise LMStudioClientError(f"An unexpected client error occurred: {e}") from e
                except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:  # Added TypeError
                    response_text = "(Could not retrieve response text)"
                    if response is not None:
                        try:
                            response_text = await response()
                        except Exception:
                            pass
                    raise LMStudioClientError(
                        f"Failed to parse LM Studio response or invalid structure: {e}. Response text: {response_text[:500]}") from e

        return LLMResult(generations=generations)

    async def _astream(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        """Async stream responses from LM Studio endpoint using aiohttp."""
        # Uses the corrected _prepare_payload
        payload = self._prepare_payload(prompt, stream=True, stop=stop, **kwargs)
        headers = self._get_headers()
        connector = aiohttp.TCPConnector(limit_per_host=100)
        session = None  # Define session/response in outer scope for potential cleanup
        response = None

        try:
            # Manage session lifetime carefully, especially in applications.
            # Creating a session per request is simpler but less efficient.
            session = aiohttp.ClientSession(connector=connector)
            response = await session.post(
                self._endpoint_url,
                headers=headers,
                json=payload,  # aiohttp handles serialization
                timeout=aiohttp.ClientTimeout(total=self.timeout)  # Use aiohttp timeout object
            )
            response.raise_for_status()  # Check initial connection

            # Process the stream line by line
            async for line_bytes in response.content:
                line = line_bytes.decode('utf-8').strip()
                if line:
                    chunk_data = _parse_sse_line(line)
                    if chunk_data:
                        if chunk_data.get("done"):
                            break
                        gen_chunk = _sse_chunk_to_generation_chunk(chunk_data)
                        yield gen_chunk
                        if run_manager:
                            # Must await callback in async context
                            await run_manager.on_llm_new_token(gen_chunk, chunk=gen_chunk)

        except aiohttp.ClientConnectorError as e:
            raise LMStudioClientError(
                f"Connection error during stream: Could not connect to {self._endpoint_url}. Is LM Studio server running? Details: {e}") from e
        except aiohttp.ClientResponseError as e:
            error_detail = "(Could not retrieve error details)"
            if response is not None:
                try:
                    error_detail = await response()
                except Exception:
                    pass
            raise LMStudioClientError(
                f"LM Studio server returned error {e.status} during stream initiation: {error_detail}"
            ) from e
        except asyncio.TimeoutError as e:
            raise LMStudioClientError(f"Request stream timed out after {self.timeout}s: {e}") from e
        except aiohttp.ClientError as e:
            raise LMStudioClientError(f"An unexpected client error occurred during streaming: {e}") from e
        except Exception as e:
            # Catch potential errors during SSE parsing or chunk processing
            import traceback
            traceback.print_exc()  # Print full traceback for debugging
            raise LMStudioClientError(f"An error occurred processing the async stream: {e}") from e
        finally:
            # Ensure response and session are closed
            if response is not None:
                response.release()  # Release connection back to pool
            if session is not None and not session.closed:
                await session.close()
