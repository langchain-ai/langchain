from __future__ import annotations
import logging
from typing import Dict, Optional
from pydantic import root_validator
from langchain.utils import get_from_dict_or_env
import requests
logger = logging.getLogger(__name__)

class EdenaiTool():
    
    feature: str
    subfeature: str
    edenai_api_key: Optional[str] = None
    provider: str
        
    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        values["edenai_api_key"] = get_from_dict_or_env(
            values, "edenai_api_key", "EDENAI_API_KEY"
        )
        return values  
    
    def _call_eden_ai(self, query_params: dict) -> list:
        
        #faire l'API call 
        
        headers = {"Authorization": f"Bearer {self.edenai_api_key}"}

        url =f"https://api.edenai.run/v2/{self.feature}/{self.subfeature}"
        
        payload={
            "providers": self.provider,
            "response_as_dict": False,
            "attributes_as_list": True,
            "show_original_response": False,
            }


        payload.update(query_params)

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
    
