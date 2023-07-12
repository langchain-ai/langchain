"""Wrapper around KoboldAI API."""
import logging
from typing import Any, Dict, List, Optional

import requests

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

logger = logging.getLogger(__name__)


def clean_url(url):
    """Remove trailing slash and /api from url if present."""
    if url.endswith('/api'):
        return url[:-4]
    elif url.endswith('/'):
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

    This will send a POST request to the Kobold API with the provided prompt and generate text.
    """
    
    endpoint: str
    """The API endpoint to use for generating text."""

    use_story: Optional[bool] = False
    """ Whether or not to use the story from the KoboldAI GUI when generating text. """

    use_authors_note: Optional[bool] = False
    """
    Whether or not to use the author's note from the KoboldAI GUI when generating text.
    This has no effect unless use_story is also enabled.
    """

    use_world_info: Optional[bool] = False
    """
    Whether or not to use the world info from the KoboldAI GUI when generating text.
    """

    use_memory: Optional[bool] = False
    """
    Whether or not to use the memory from the KoboldAI GUI when generating text.
    """

    max_context_length: Optional[int] = 1600
    """
    minimum: 1
    Maximum number of tokens to send to the model.
    """

    max_length: Optional[int] = 80
    """
    maximum: 512
    minimum: 1
    Number of tokens to generate.

    """

    rep_pen: Optional[float] = 1.12
    """
    Base repetition penalty value.
    minimum: 1
    """

    rep_pen_range: Optional[int] = 1024
    """

    Repetition penalty range.
    minimum: 0

    """

    rep_pen_slope: Optional[float] = 0.9
    """
    minimum: 0
    Repetition penalty slope.

    """

    temperature: Optional[float] = 0.6
    """
    exclusiveMinimum: 0

    Temperature value.
    """

    tfs: Optional[float] = 0.9
    """
    maximum: 1
    minimum: 0
    Tail free sampling value.
    """

    top_a: Optional[float] = 0.9
    """
    minimum: 0
    Top-a sampling value.
    """

    top_p: Optional[float] = 0.95
    """
    maximum: 1
    minimum: 0
    Top-p sampling value.
    """

    top_k: Optional[int] = 0
    """
    minimum: 0
    Top-k sampling value.
    """

    typical: Optional[float] = 0.5
    """
    maximum: 1
    minimum: 0
    Typical sampling value.
    """

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]]=None,
        run_manager: Optional[CallbackManagerForLLMRun]=None,
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
        data = {
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
        """
        Add the stop sequences to the data if they are provided
        maxItems: 10
        An array of string sequences where the API will stop generating further tokens. The returned text WILL contain the stop sequence.
        """
            
        response = requests.post(f"{clean_url(self.endpoint)}/api/v1/generate", json=data)
        """
        Send a POST request to the API endpoint
        """

        response.raise_for_status()
        """
        Raise an exception if the response is not successful
        """

        json_response = response.json()
        """
        Parse the response as JSON
        """

        if "results" in json_response and len(json_response["results"]) > 0 and "text" in json_response["results"][0]:
            text = json_response["results"][0]["text"].strip()
            """
            Remove leading and trailing whitespace from the text
            """

            if stop is not None:
                for sequence in stop:
                    if text.endswith(sequence):
                        text = text[:-len(sequence)].rstrip()
            """
            Remove the stop sequence from the end of the text, if it's there
            """
            
            return text
        else:
            raise ValueError("Unexpected response format from Kobold API")
        """
        Return the generated text
        """
