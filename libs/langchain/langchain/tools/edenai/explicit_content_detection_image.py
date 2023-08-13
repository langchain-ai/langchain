from __future__ import annotations
import logging
from typing import Dict, Optional
from pydantic import root_validator
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.utils import get_from_dict_or_env
from langchain.tools.edenai.edenai_base_tool import EdenaiTool
logger = logging.getLogger(__name__)
  

class EdenAiExplicitImage(EdenaiTool):
    
    """Tool that queries the Eden AI Explicit image detection.

    for api reference check edenai documentation: https://docs.edenai.co/reference/image_explicit_content_create.
    
    To use, you should have
    the environment variable ``EDENAI_API_KEY`` set with your API token.
    You can find your token here: https://app.edenai.run/admin/account/settings

    """
    edenai_api_key: Optional[str] = None

    name="edenai_image_explicit_content_detection"
    
    description = (
        "A wrapper around edenai Services Explicit image detection. "
        """Useful for when you have to extract Explicit Content Detection detects adult only content in images, 
        that is generally inappropriate for people under
        the age of 18 and includes nudity, sexual activity, pornography, violence, gore content, etc."""
        "Input should be the string url of the image ."
    )
    
    url="https://api.edenai.run/v2/image/explicit_content"
        
    provider: str
    """ provider to use"""
    
    feature="image"
    subfeature="explicit_content"
    
    
    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        values["edenai_api_key"] = get_from_dict_or_env(
            values, "edenai_api_key", "EDENAI_API_KEY"
        )
        return values
    
    def _format_json_to_string(self,json_data: list) -> str:
        for item in json_data:
            if item["provider"]== "eden-ai" :
                result_str = f"nsfw_likelihood: {item['nsfw_likelihood']}\n"
                for idx, item in enumerate(item["items"]):
                    label = item["label"].lower()
                    likelihood = item["likelihood"]
                    result_str += f"{idx}: {label} likelihood {likelihood},\n"
            else : 
                pass
            
        return result_str[:-2]

        
    
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        try:
            query_params = {"file_url": query,"attributes_as_list": False}
            text_analysis_result = self._call_eden_ai(query_params)
            text_analysis_result=text_analysis_result.json()
            return self._format_json_to_string(text_analysis_result)

        except Exception as e:
            raise RuntimeError(f"Error while running EdenAiExplicitText: {e}")
