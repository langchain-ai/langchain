from __future__ import annotations
import logging
from typing import Dict, Optional,Any
from pydantic import root_validator,Field
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.edenai.edenai_base_tool import EdenaiTool
from langchain.utils import get_from_dict_or_env
logger = logging.getLogger(__name__)


class EdenAiObjectDetectionTool(EdenaiTool):
    """Tool that queries the Eden AI Object detection API.

    for api reference check edenai documentation: https://docs.edenai.co/reference/image_object_detection_create.
    
    To use, you should have
    the environment variable ``EDENAI_API_KEY`` set with your API token.
    You can find your token here: https://app.edenai.run/admin/account/settings

    """
    edenai_api_key: Optional[str] = None

    name="edenai_object_detection"
    
    description = (
        "A wrapper around edenai Services Object Detection . "
       "Useful for when you have to do an  to identify and locate (with bounding boxes) objects in an image "
        "Input should be the string url of the image to identify."
    )
    
    base_url = "https://docs.edenai.co/reference/image_object_detection_create"

    params : Optional[Dict[str,Any]] = Field(default_factory=dict)
    
    show_positions : bool = False
    
    feature="image"
    subfeature = "object_detection"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        values["edenai_api_key"] = get_from_dict_or_env(
            values, "edenai_api_key", "EDENAI_API_KEY"
        )
        return values
    
    
    def _parse_json(self,json_data : dict ) -> str :       
        result = []
        label_info = []
    
        for found_obj in json_data['items']:
            label_str = f"{found_obj['label']} - Confidence {found_obj['confidence']}"
            x_min = found_obj.get("x_min")
            x_max = found_obj.get("x_max")
            y_min = found_obj.get("y_min")
            y_max = found_obj.get("y_max")
            if self.show_positions and all([x_min, x_max, y_min, y_max]):  # some providers don't return positions
                label_str += f",at the position x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max: {y_max}"  
            label_info.append(label_str)
    
        result.append("\n".join(label_info))
        return "\n\n".join(result)
            
    
    def _format_object_detection_result(self,json_data : list )->str:
        if len(json_data) == 1 :
            result=self._parse_json(json_data[0])
        else:
            for entry in json_data:
                if entry.get("provider") == "eden-ai":                    
                    result=self._parse_json(entry)
    
        return result            
        

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        try:
            query_params = {"file_url": query,"attributes_as_list": False}
            image_analysis_result = self._call_eden_ai(query_params)
            image_analysis_result=image_analysis_result.json()
            return self._format_object_detection_result(image_analysis_result)

        except Exception as e:
            raise RuntimeError(f"Error while running EdenAiExplicitText: {e}")



