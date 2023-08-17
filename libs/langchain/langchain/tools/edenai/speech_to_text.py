from __future__ import annotations
import logging
from typing import Dict, Optional,Any
from pydantic import root_validator,Field
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.utils import get_from_dict_or_env
from langchain.tools.edenai import EdenaiTool
import json
logger = logging.getLogger(__name__)
  

logger = logging.getLogger(__name__)

class EdenAiSpeechToText(EdenaiTool):
    """Tool that queries the Eden AI Speech To Text API.

    for api reference check edenai documentation: https://app.edenai.run/bricks/speech/asynchronous-speech-to-text.
    
    To use, you should have
    the environment variable ``EDENAI_API_KEY`` set with your API token.
    You can find your token here: https://app.edenai.run/admin/account/settings

    """
    edenai_api_key : Optional[str] = None  

    name="edenai_speech_to_text"
    description = (
        "A wrapper around edenai Services speech to text "
        "Useful for when you have to convert audio to text."
        "Input should be a url to an audio file."
    )
    
    base_url = "https://api.edenai.run/v2/audio/speech_to_text_async"
    
    
    language: Optional[str] ="en"
    params : Optional[Dict[str,Any]] = Field(default_factory=dict)
    
    feature : str = "audio"
    subfeature: str = "speech_to_text_async"

    base_url="https://api.edenai.run/v2/audio/speech_to_text_async/"          

    def _format_text_explicit_content_detection_result(
        self,
        text_analysis_result: list) -> str:
        pass
    
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        try:
            query_params = {"file_url": query, "language": self.language ,**self.params}
            get_job_result = self._call_eden_ai(query_params)
            
            job_id=get_job_result.json()["public_id"]
            
            url=self.base_url+job_id
            
            audio_analysis_result = self._get_edenai(url)
            
            result=audio_analysis_result.text 
            formatted_text=json.loads(result)

            text=formatted_text['results'][self.provider]['text']

            return text


        except Exception as e:
            raise RuntimeError(f"Error while running EdenAiExplicitText: {e}")
