"""Wrapper around KoboldAI API."""
import logging
from typing import Any, Dict, List, Optional

import requests

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

logger = logging.getLogger(__name__)


def clean_url(url: str) -> str:
    """Remove trailing slash and /api from url if present."""
    if url.endswith("/api"):
        return url[:-4]
    elif url.endswith("/"):
        return url[:-1]
    else:
        return url


class KoboldApiLLM(LLM):
    """
    A class that acts as a wrapper for the Kobold API language model.

    It includes several fields that can be used to control the text generation process.

    To use this class, instantiate it with the required parameters and call it with a
    prompt to generate text. For example:

        kobold = KoboldApiLLM(endpoint="http://localhost:5000")
        result = kobold("Write a story about a dragon.")

    This will send a POST request to the Kobold API with the provided prompt and
    generate text.
    """

    endpoint: str
    """The API endpoint to use for generating text."""

    use_story: Optional[bool] = False
    """ Whether or not to use the story from the KoboldAI GUI when generating text. """

    use_authors_note: Optional[bool] = False
    """Whether to use the author's note from the KoboldAI GUI when generating text.
    
    This has no effect unless use_story is also enabled.
    """

    use_world_info: Optional[bool] = False
    """Whether to use the world info from the KoboldAI GUI when generating text."""

    use_memory: Optional[bool] = False
    """Whether to use the memory from the KoboldAI GUI when generating text."""

    max_context_length: Optional[int] = 1600
    """Maximum number of tokens to send to the model.
    
    minimum: 1
    """

    max_length: Optional[int] = 80
    """Number of tokens to generate.
    
    maximum: 512
    minimum: 1
    """

    rep_pen: Optional[float] = 1.12
    """Base repetition penalty value.
    
    minimum: 1
    """

    rep_pen_range: Optional[int] = 1024
    """Repetition penalty range.
    
    minimum: 0
    """

    rep_pen_slope: Optional[float] = 0.9
    """Repetition penalty slope.
    
    minimum: 0
    """

    temperature: Optional[float] = 0.6
    """Temperature value.
    
    exclusiveMinimum: 0
    """

    tfs: Optional[float] = 0.9
    """Tail free sampling value.
    
    maximum: 1
    minimum: 0
    """

    top_a: Optional[float] = 0.9
    """Top-a sampling value.
    
    minimum: 0
    """

    top_p: Optional[float] = 0.95
    """Top-p sampling value.
    
    maximum: 1
    minimum: 0
    """

    top_k: Optional[int] = 0
    """Top-k sampling value.
    
    minimum: 0
    """

    typical: Optional[float] = 0.5
    """Typical sampling value.
    
    maximum: 1
    minimum: 0
    """

    @property
    def _llm_type(self) -> str:
        return "koboldai"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the API and return the output.

        Args:
            prompt: The prompt to use for generation.
            stop: A list of strings to stop generation when encountered.

        Returns:
            The generated text.

        Example:
            .. code-block:: python

                from langchain.llms import KoboldApiLLM

                llm = KoboldApiLLM(endpoint="http://localhost:5000")
                llm("Write a story about dragons.")
        """
        data: Dict[str, Any] = {
            "prompt": prompt,
            "use_story": self.use_story,
            "use_authors_note": self.use_authors_note,
            "use_world_info": self.use_world_info,
            "use_memory": self.use_memory,
            "max_context_length": self.max_context_length,
            "max_length": self.max_length,
            "rep_pen": self.rep_pen,
            "rep_pen_range": self.rep_pen_range,
            "rep_pen_slope": self.rep_pen_slope,
            "temperature": self.temperature,
            "tfs": self.tfs,
            "top_a": self.top_a,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "typical": self.typical,
        }

        if stop is not None:
            data["stop_sequence"] = stop

        response = requests.post(
            f"{clean_url(self.endpoint)}/api/v1/generate", json=data
        )

        response.raise_for_status()
        json_response = response.json()

        if (
            "results" in json_response
            and len(json_response["results"]) > 0
            and "text" in json_response["results"][0]
        ):
            text = json_response["results"][0]["text"].strip()

            if stop is not None:
                for sequence in stop:
                    if text.endswith(sequence):
                        text = text[: -len(sequence)].rstrip()

            return text
        else:
            raise ValueError(
                f"Unexpected response format from Kobold API:  {json_response}"
            )
