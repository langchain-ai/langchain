from __future__ import annotations

import logging
from typing import Optional

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.edenai.edenai_base_tool import EdenaiTool

logger = logging.getLogger(__name__)


class EdenAiExplicitImageTool(EdenaiTool):

    """Tool that queries the Eden AI Explicit image detection.

    for api reference check edenai documentation:
    https://docs.edenai.co/reference/image_explicit_content_create.

    To use, you should have
    the environment variable ``EDENAI_API_KEY`` set with your API token.
    You can find your token here: https://app.edenai.run/admin/account/settings

    """

    edenai_api_key: Optional[str] = None

    name = "edenai_image_explicit_content_detection"

    description = (
        "A wrapper around edenai Services Explicit image detection. "
        """Useful for when you have to extract Explicit Content from images.
        it detects adult only content in images, 
        that is generally inappropriate for people under
        the age of 18 and includes nudity, sexual activity,
        pornography, violence, gore content, etc."""
        "Input should be the string url of the image ."
    )

    feature = "image"
    subfeature = "explicit_content"

    def _parse_json(self, json_data: dict) -> str:
        result_str = f"nsfw_likelihood: {json_data['nsfw_likelihood']}\n"
        for idx, found_obj in enumerate(json_data["items"]):
            label = found_obj["label"].lower()
            likelihood = found_obj["likelihood"]
            result_str += f"{idx}: {label} likelihood {likelihood},\n"

        return result_str[:-2]

    def _format_explicit_image(self, json_data: list) -> str:
        if len(json_data) == 1:
            result = self._parse_json(json_data[0])
        else:
            for entry in json_data:
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
        image_analysis_result = self._call_eden_ai(query_params)
        try:
            image_analysis_dict = image_analysis_result.json()
            return self._format_explicit_image(image_analysis_dict)

        except Exception as e:
            raise RuntimeError(f"Error while running EdenAiExplicitText: {e}")
