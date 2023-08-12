from __future__ import annotations
import logging
from typing import Dict, Optional
from pydantic import root_validator
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.base import BaseTool
from langchain.utils import get_from_dict_or_env
import requests
logger = logging.getLogger(__name__)

class EdenAiParsingIDTool(BaseTool):
    """Tool that queries the Eden AI  Identity parsingAPI.

    for api reference check edenai documentation: https://docs.edenai.co/reference/ocr_identity_parser_create.
    
    To use, you should have
    the environment variable ``EDENAI_API_KEY`` set with your API token.
    You can find your token here: https://app.edenai.run/admin/account/settings

    """
    edenai_api_key: Optional[str] = None

    name="edenai_identity_parsing"
    
    description = (
        "A wrapper around edenai Services Identity parsing. "
        "Useful for when you have to extract information from an ID Document "
        "Input should be the string url of the document to parse."
    )
    
    base_url = "https://api.edenai.run/v2/ocr/identity_parser"
    
    language: Optional[str] = None
    """
    language of the text passed to the model.
    """    
    
    provider: str
    """ provider to use (amazon,base64,microsoft,mindee,klippa )"""
    params : Optional[Dict[str,any]] = None

    
    
    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        values["edenai_api_key"] = get_from_dict_or_env(
            values, "edenai_api_key", "EDENAI_API_KEY"
        )
        return values
    
    def _format_value(self,value):
        if isinstance(value, dict):
            if "value" in value:
                return value["value"]
            elif "name" in value:
                return value["name"]
        elif isinstance(value, list):
            return " ".join([self.format_value(item) for item in value])
        return value

    
    
    def _format_id_parsing_result(self, id_parsing_result: list) -> str:
        formatted_text = ""
        
        extracted_data = id_parsing_result["extracted_data"][0]
        for key, value in extracted_data.items():
            formatted_value = self._format_value(value)
            formatted_text += f"{key} : {formatted_value}\n"
        
        return formatted_text
        
    
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        try:
            query_params = {"file_url": query, "language": self.language,**self.params}
            text_analysis_result = self._call_eden_ai(query_params)
            text_analysis_result=text_analysis_result.json()
            return self._format_id_parsing_result(text_analysis_result)

        except Exception as e:
            raise RuntimeError(f"Error while running EdenAiExplicitText: {e}")



