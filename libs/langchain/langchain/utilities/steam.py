"""Util that calls Steam-WebAPI."""

from langchain.pydantic_v1 import BaseModel, Extra, root_validator
from typing import Any, Dict, List, Optional
from langchain.langchain.tools.steam.prompt import STEAM_GET_GAMES_ID, STEAM_GET_GAMES_DETAILS
from langchain.utils import get_from_dict_or_env  
import steamspypi

class SteamWebAPIWrapper(BaseModel):
    # Steam WebAPI Implementation will go here...

    steam: Any  #for python-steam-api

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
            from decouple import config
        except ImportError:
            raise ImportError(
                "decouple library is not installed. "
              
            )
        
        #initilize the steam attribute for python-steam-api usage
        KEY = config(values["steam_key"])
        steam = Steam(KEY)
        return values




    def parse_to_str(self, details: Dict) -> str: #NOT SURE IF details IS A DICT OF LIST OF DICT
        """Parse the details result."""
        result=""
        for key, value in details.items():
            result+= str(key) + '->' + str(value) + '\n'
        return result

    def get_id(self, games: Dict, name: str) -> str:
        """ The response may contain more than one game, so we need to choose the right one and 
        return the id."""

        for app in games.get("apps", []):
            if app["name"].lower() == name.lower():
                return str(app["id"])

    def details_of_games(self, name: str) -> str:   
        
        #get id
        games = self.steam.apps.search_games(name)
        id = self.get_id(games, name)

        #use id to get details
        data_request = dict()
        data_request['request'] = 'appdetails'
        data_request['appid'] = id
        data = steamspypi.download(self.data_request)
        parsed_data = self.parse_to_str(data)
        return parsed_data


   
    def run(self, mode: str, game:str) -> str:

        if mode == "get_game_ID":
            return self.get_id(game)
        elif mode == "get_game_Details":
            return self.details_of_games(game)
        else:
            raise ValueError(f"Invalid mode {mode} for Steam API.")
