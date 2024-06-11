from __future__ import annotations

import logging
from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun

from langchain_community.tools.edenai.edenai_base_tool import EdenaiTool

logger = logging.getLogger(__name__)


class EdenAiObjectDetectionTool(EdenaiTool):
    """Tool that queries the Eden AI Object detection API.

    for api reference check edenai documentation:
    https://docs.edenai.co/reference/image_object_detection_create.

    To use, you should have
    the environment variable ``EDENAI_API_KEY`` set with your API token.
    You can find your token here: https://app.edenai.run/admin/account/settings

    """

    name: str = "edenai_object_detection"

    description: str = (
        "A wrapper around edenai Services Object Detection . "
        """Useful for when you have to do an  to identify and locate
        (with bounding boxes) objects in an image """
        "Input should be the string url of the image to identify."
    )

    show_positions: bool = False

    feature: str = "image"
    subfeature: str = "object_detection"

    def _parse_json(self, json_data: dict) -> str:
        result = []
        label_info = []

        for found_obj in json_data["items"]:
            label_str = f"{found_obj['label']} - Confidence {found_obj['confidence']}"
            x_min = found_obj.get("x_min")
            x_max = found_obj.get("x_max")
            y_min = found_obj.get("y_min")
            y_max = found_obj.get("y_max")
            if self.show_positions and all(
                [x_min, x_max, y_min, y_max]
            ):  # some providers don't return positions
                label_str += f""",at the position x_min: {x_min}, x_max: {x_max}, 
                y_min: {y_min}, y_max: {y_max}"""
            label_info.append(label_str)

        result.append("\n".join(label_info))
        return "\n\n".join(result)

    def _parse_response(self, response: list) -> str:
        if len(response) == 1:
            result = self._parse_json(response[0])
        else:
            for entry in response:
                if entry.get("provider") == "eden-ai":
                    result = self._parse_json(entry)

        return result

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        query_params = {"file_url": query, "attributes_as_list": False}
        return self._call_eden_ai(query_params)
