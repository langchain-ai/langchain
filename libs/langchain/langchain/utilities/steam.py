"""Util that calls Steam-WebAPI."""

from langchain.pydantic_v1 import BaseModel, Extra, root_validator
from typing import Any, Dict, List, Optional
from langchain.langchain.tools.steam.prompt import STEAM_GET_GAMES_ID, STEAM_GET_GAMES_DETAILS, STEAM_GET_RECOMMENDED_GAMES
from langchain.utils import get_from_dict_or_env  
import steamspypi
import json

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
        {
            "mode": "get_recommended_GAMES",
            "name": "Get Recommended Games",
            "description": STEAM_GET_RECOMMENDED_GAMES,
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
        values["steam"] = steam
        return values

    def parse_to_str(self, details: Dict) -> str: #NOT SURE IF details IS A DICT OF LIST OF DICT
        """Parse the details result."""
        result=""
        for key, value in details.items():
            result+= "The" + str(key) + 'is: ' + str(value) + '\n'
        return result
    


    def get_id(self, games: Dict, name: str) -> Dict:
        """ The response may contain more than one game, so we need to choose the right one and 
        return the id."""

        game_info = {}
        for app in games['apps']:
            if app['name'].lower() == name.lower():
                game_info['id'] = app['id']
                game_info['link'] = app['link']
                game_info['price'] = app['price']
                break
        return game_info
            


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
    
    ##############################################################################################################
    
    # get steam id from username
    def get_steam_id(self, name: str) -> str:
        user_json = self.steam.users.search_user(name)
        user = json.loads(user_json)
        steamId = user['player']['steamid']
        return steamId

    def recommended_games(self, name: str) -> str:
        steam_id = self.get_steam_id(name)
        user_games_json = self.steam.users.get_owned_games(steam_id)
        user_games_data = json.loads(user_games_json)

        appids_with_playtime = [(game['appid'], game.get('playtime_forever', 0)) for game in user_games_data['response']['games']]
        sorted_appids = sorted(appids_with_playtime, key=lambda x: x[1], reverse=True)
        
        app_details = []
        for app_id, _ in sorted_appids:
            app_detail = self.steam.apps.get_app_details(app_id)
            app_details.append(app_detail)
        
        #TODO: get recommended games using langchain
   
    def run(self, mode: str, game:str) -> str:

        if mode == "get_game_ID":
            return self.get_id(game)
        elif mode == "get_game_Details":
            return self.details_of_games(game)
        elif mode == "get_recommended_games":
            return self.recommended_games(game)
        else:
            raise ValueError(f"Invalid mode {mode} for Steam API.")
