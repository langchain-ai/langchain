"""Util that calls several NASA APIs."""
import json

import requests

from langchain.pydantic_v1 import BaseModel

IMAGE_AND_VIDEO_LIBRARY_URL = "https://images-api.nasa.gov"


class NasaAPIWrapper(BaseModel):
    def get_media(self, query: str) -> str:
        params = json.loads(query)
        if params.get("q"):
            queryText = params["q"]
            params.pop("q")
        else:
            queryText = ""
        response = requests.get(
            IMAGE_AND_VIDEO_LIBRARY_URL + "/search?q=" + queryText, params=params
        )
        data = response.json()
        return data

    def get_media_metadata_manifest(self, query: str) -> str:
        response = requests.get(IMAGE_AND_VIDEO_LIBRARY_URL + "/asset/" + query)
        return response.json()

    def get_media_metadata_location(self, query: str) -> str:
        response = requests.get(IMAGE_AND_VIDEO_LIBRARY_URL + "/metadata/" + query)
        return response.json()

    def get_video_captions_location(self, query: str) -> str:
        response = requests.get(IMAGE_AND_VIDEO_LIBRARY_URL + "/captions/" + query)
        return response.json()

    def run(self, mode: str, query: str) -> str:
        if mode == "search_media":
            output = self.get_media(query)
        elif mode == "get_media_metadata_manifest":
            output = self.get_media_metadata_manifest(query)
        elif mode == "get_media_metadata_location":
            output = self.get_media_metadata_location(query)
        elif mode == "get_video_captions_location":
            output = self.get_video_captions_location(query)
        else:
            output = f"ModeError: Got unexpected mode {mode}."

        try:
            return json.dumps(output)
        except Exception:
            return str(output)
