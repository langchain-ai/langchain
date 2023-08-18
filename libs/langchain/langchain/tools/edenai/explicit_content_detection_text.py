from __future__ import annotations
import logging
from typing import Dict, Optional
from pydantic import root_validator
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.base import BaseTool
from langchain.utils import get_from_dict_or_env
from langchain.tools.edenai.edenai_base_tool import EdenaiTool
logger = logging.getLogger(__name__)
  
class EdenAiExplicitTextDetection(EdenaiTool):  
    """Tool that queries the Eden AI Explicit text detection.

    for api reference check edenai documentation: https://docs.edenai.co/reference/image_explicit_content_create.
    
    To use, you should have
    the environment variable ``EDENAI_API_KEY`` set with your API token.
    You can find your token here: https://app.edenai.run/admin/account/settings

    """
    
    edenai_api_key : Optional[str] = None  

    name="edenai_explicit_content_detection_text"
    
    description = (
        "A wrapper around edenai Services explicit content detection for text. "
        
        """Useful for when you have to scan text for offensive, sexually explicit or suggestive content,
        it checks also if there is any content of self-harm, violence, racist or hate speech."""

        """the structure of the observation is : 
        'the type of explicity : the likelihood of it being explicit'
        consider something is explicit if the likelihood is equal or above 3 
        """
        
        "Input should be a string."
    )
    
    language: str

    feature : str = "text"
    subfeature: str = "moderation"


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
            query_params = {"text": query, "language": self.language}
            text_analysis_result = self._call_eden_ai(query_params)
            text_analysis_result=text_analysis_result.json()
            return self._format_text_explicit_content_detection_result(text_analysis_result)

        except Exception as e:
            raise RuntimeError(f"Error while running EdenAiExplicitText: {e}")
