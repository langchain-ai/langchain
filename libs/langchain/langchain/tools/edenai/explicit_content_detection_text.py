from __future__ import annotations
import logging
from typing import Dict, Optional
from pydantic import root_validator
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.base import BaseTool
from langchain.utils import get_from_dict_or_env
import requests
logger = logging.getLogger(__name__)

class EdenAiExplicitTextDetection(BaseTool):
    """Tool that queries the Eden AI Explicit content detection API.

    for api reference check edenai documentation: https://docs.edenai.co/reference/text_moderation_create.
    
    To use, you should have
    the environment variable ``EDENAI_API_KEY`` set with your API token.
    You can find your token here: https://app.edenai.run/admin/account/settings

    """
    edenai_api_key: Optional[str] = None

    name="edenai_explicit_content_detection_text"
    
    description = (
        "A wrapper around edenai Services explicit content detection for text. "
        "Useful for when you have to scan text for offensive, sexually explicit or suggestive content, it checks also if there is any content of self-harm, violence, racist or hate speech."
        "Input should be a string."
    )
    
    base_url = "https://api.edenai.run/v2/text/moderation"
    
    language: Optional[str] = None
    """
    language of the text passed to the model.
    """    
    
    provider: str
    """ provider to use (eg: openai,clarifai, etc.)"""

    
    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        values["edenai_api_key"] = get_from_dict_or_env(
            values, "edenai_api_key", "EDENAI_API_KEY"
        )
        return values     
    
    def _text_explicit_content_detection(self, text: str) -> list:
        
        #faire l'API call 
        
        headers = {"Authorization": f"Bearer {self.edenai_api_key}"}

        url ="https://api.edenai.run/v2/text/moderation"
        
        payload={
            "providers": self.provider,
            "text": text,
            "response_as_dict": False,
            "attributes_as_list": True,
            "show_original_response": False,
            "language": self.language,
            }

        response = requests.post(url, json=payload, headers=headers)

        if response.status_code >= 500:
            raise Exception(f"EdenAI Server: Error {response.status_code}")
        elif response.status_code >= 400:
            raise ValueError(f"EdenAI received an invalid payload: {response.text}")
        elif response.status_code != 200:
            raise Exception(
                f"EdenAI returned an unexpected response with status "
                f"{response.status_code}: {response.text}"
            )
            
        if "error" in response.json()[0].keys():
            raise ValueError(f"EdenAI received an invalid payload: {response.json()[0]['error']['message'] }" )
        
        return response 
    
    def _format_text_explicit_content_detection_result(
        self,
        text_analysis_result: list) -> str:
        formatted_result = []
        for result in text_analysis_result:
            if "nsfw_likelihood" in result.keys():
                formatted_result.append("nsfw_likelihood: " + str(result["nsfw_likelihood"]))
                
            for label, likelihood in zip(result['label'], result['likelihood']):
                formatted_result.append(f'"{label}": {str(likelihood)}')

        return "\n".join(formatted_result)

    
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        try:
            text_analysis_result = self._text_explicit_content_detection(query)
            text_analysis_result=text_analysis_result.json()
            return self._format_text_explicit_content_detection_result(text_analysis_result)
        
        except Exception as e:
            raise RuntimeError(f"Error while running EdenAiExplicitText: {e}")
     
