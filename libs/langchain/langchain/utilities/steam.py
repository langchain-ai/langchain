"""Util that calls Steam-WebAPI."""

from langchain.pydantic_v1 import BaseModel, Extra, root_validator
from typing import Any, Dict, List, Optional
from langchain.langchain.tools.steam.prompt import STEAM_GET_GAMES_ID, STEAM_GET_GAMES_DETAILS
from langchain.utils import get_from_dict_or_env  


class SteamWebAPIWrapper(BaseModel):
    # Steam WebAPI Implementation will go here...

    steam: Any  #for python-steam-api
    data_request: Any #for steamspypi

    # can decide later if needed

    #oprations: a list of dictionaries, each representing a specific operation that can be performed with the API
    operations: List[Dict]=[
      

        { 
            "mode": "get_game_ID",
            "name": "Get Game ID",
            "description": STEAM_GET_GAMES_ID, 
        },

        {
            "mode": "get_game_Details",
            "name": "Get Game Details",
            "description": STEAM_GET_GAMES_DETAILS,
        },
    ]



    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def list(self) ->List[Dict]:
        """Return a list of operations."""
        return self.operations
    

    @root_validator
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate api key and python package has been configed."""

        steam_key = get_from_dict_or_env(values, "steam_key", "STEAM_KEY") #look for a value in two places: values dictionary/environment variables of the operating system 
       
        #if it's in env, then added to values dictionary
        values["steam_key"] = steam_key  

        # check if the python package is installed
        try:
            from steam import Steam  
        except ImportError:
            raise ImportError(
                "python-steam-api library is not installed. "
              
            )
       
        try:
            import steamspypi
        except ImportError:
            raise ImportError(
                "steamspypi library is not installed. "
            )

       ################################################################################# 
        #NOT SURE IF WE INITILAIZE THE steam and data_request HERE OR LATER!!!
       #################################################################################
        return values





    def run(self, prompt: str = "demo") -> str:
        return prompt
