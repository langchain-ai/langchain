"""Util that calls several NASA APIs."""
import json
from dataclasses import asdict, dataclass, fields

import requests

from langchain.pydantic_v1 import BaseModel

IMAGE_AND_VIDEO_LIBRARY_URL = "https://images-api.nasa.gov"
EXOPLANETS_URL = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?"


class NasaAPIWrapper(BaseModel):

    def get_media(self, query: str) -> str:
        params = json.loads(query)
        queryText = params['q']
        params.pop('q')
        response = requests.get(IMAGE_AND_VIDEO_LIBRARY_URL + "/search?q=" + queryText, params=params)
        data = response.json()
        return data
    
    def get_media_metadata_manifest(self, query: str) -> str:
        response = requests.get(IMAGE_AND_VIDEO_LIBRARY_URL + "/assets/" + query)
        return response.json()
    
    def get_media_metadata_location(self, query: str) -> str:
        response = requests.get(IMAGE_AND_VIDEO_LIBRARY_URL + "/metadata/" + query)
        return response.json()
        
    def get_video_captions_location(self, query: str) -> str:
        response = requests.get(IMAGE_AND_VIDEO_LIBRARY_URL + "/captions/" + query)
        return response.json()
    


    def get_expoplanet_info(self, query: str) -> str:
        params = json.loads(query)
        response = requests.get(EXOPLANETS_URL, params=params)
        data = response.json()
        return data
    

    def run(self, mode: str, query: str) -> str:
        if mode == 'get_media':
            output = self.get_media(query)
        elif mode == 'get_media_metadata_manifest':
            output = self.get_media_metadata_manifest(query)
        elif mode == 'get_media_metadata_location':
            output = self.get_media_metadata_location(query)
        elif mode == 'get_video_captions_location':
            output = self.get_video_captions_location(query)
        elif mode == 'exoplanet':
            output = self.get_expoplanet_info(query)
        else:
            output = {"ModeError": f"Got unexpected mode {mode}."}

        try:
            return json.dumps(output)
        except Exception:
            return str(output)
