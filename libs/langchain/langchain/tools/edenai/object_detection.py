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
    
    provider: str
    """ provider to use (amazon,base64,microsoft,mindee,klippa )"""
    
    params : Optional[Dict[str,Any]] = Field(default_factory=dict)
    
    show_positions : bool = True
    
    feature="image"
    subfeature = "object_detection"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        values["edenai_api_key"] = get_from_dict_or_env(
            values, "edenai_api_key", "EDENAI_API_KEY"
        )
        return values
    
    
    def _format_object_detection_result(self,json_data : list )->str:
        result = []
        for entry in json_data:
            if entry.get("provider") == "eden-ai":
                label_info = []
                for label, confidence in zip(entry.get("label", []), entry.get("confidence", [])):
                    label_str = f"{label} - Confidence: {confidence}"
                    if self.show_positions:
                        x_min = entry.get("x_min", [])[entry.get("label", []).index(label)]
                        x_max = entry.get("x_max", [])[entry.get("label", []).index(label)]
                        y_min = entry.get("y_min", [])[entry.get("label", []).index(label)]
                        y_max = entry.get("y_max", [])[entry.get("label", []).index(label)]
                        label_str += f",at the position x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max: {y_max}"
                    label_info.append(label_str)
            else:
                pass

        
        result.append("\n".join(label_info))

        return "\n\n".join(result)
    
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        try:
            query_params = {"file_url": query}
            image_analysis_result = self._call_eden_ai(query_params)
            image_analysis_result=image_analysis_result.json()
            return self._format_object_detection_result(image_analysis_result)

        except Exception as e:
            raise RuntimeError(f"Error while running EdenAiExplicitText: {e}")



