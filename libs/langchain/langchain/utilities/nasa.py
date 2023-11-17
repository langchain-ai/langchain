"""Util that calls several NASA APIs."""
import json
import warnings
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type, Union

import requests

from langchain.pydantic_v1 import BaseModel, Extra, root_validator
from langchain.utils import get_from_dict_or_env

IMAGE_AND_VIDEO_LIBRARY_URL = "https://images-api.nasa.gov"
EXOPLANETS_URL = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?"


class NasaAPIWrapper(BaseModel):

    def get_media(self, query: str) -> str:
        params = json.loads(query)
        #params needs to have a query q, and optionally a bunch of params
        response = requests.get(IMAGE_AND_VIDEO_LIBRARY_URL + "/search?q=" + params['q'], params=params)
        data = response.json()
        return data
    


    def get_expoplanet_info(self, query: str) -> str:
        params = json.loads(query)
        response = requests.get(EXOPLANETS_URL, params=params)
        data = response.json()
        return data
    

    def run(self, mode: str, query: str) -> str:
        if mode == 'get_media':
            output = self.get_media(query)
        elif mode == 'exoplanet':
            output = self.get_expoplanet_info(query)
        else:
            output = {"ModeError": f"Got unexpected mode {mode}."}

        try:
            return json.dumps(output)
        except Exception:
            return str(output)
