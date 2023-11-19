"""Util that calls Steam-WebAPI."""

from langchain.pydantic_v1 import BaseModel, Extra, root_validator
from typing import Any, dict
from langchain.langchain.tools.steam.prompt import (
    STEAM_GET_GAMES_ID,
    STEAM_GET_GAMES_DETAILS,
    STEAM_GET_RECOMMENDED_GAMES,
)
from langchain.utils import get_from_dict_or_env
import steamspypi
import json
from bs4 import BeautifulSoup


class SteamWebAPIWrapper(BaseModel):
    # Steam WebAPI Implementation will go here...

    steam: Any  # for python-steam-api

    # oprations: a list of dictionaries, each representing a specific operation that can be performed with the API
    operations: list[dict] = [
        {
            "mode": "get_games_id",
            "name": "Get Games ID",
            "description": STEAM_GET_GAMES_ID,
        },
        {
            "mode": "get_game_details",
            "name": "Get Game Details",
            "description": STEAM_GET_GAMES_DETAILS,
        },
        {
            "mode": "get_recommended_games",
            "name": "Get Recommended Games",
            "description": STEAM_GET_RECOMMENDED_GAMES,
        },
    ]

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def get_operations(self) -> list[dict]:
        """Return a list of operations."""
        return self.operations

    @root_validator
    def validate_environment(cls, values: dict) -> dict:
        """Validate api key and python package has been configed."""

        steam_key = get_from_dict_or_env(
            values, "steam_key", "STEAM_KEY"
        )  # look for a value in two places: values dictionary/environment variables of the operating system

        # if it's in env, then added to values dictionary
        values["steam_key"] = steam_key

        # check if the python package is installed
        try:
            from steam import Steam
        except ImportError:
            raise ImportError("python-steam-api library is not installed. ")

        try:
            from decouple import config
        except ImportError:
            raise ImportError("decouple library is not installed. ")

        # initilize the steam attribute for python-steam-api usage
        KEY = config(values["steam_key"])
        steam = Steam(KEY)
        values["steam"] = steam
        return values

    def parse_to_str(self, details: dict) -> str:  # For later parsing
        """Parse the details result."""
        result = ""
        for key, value in details.items():
            result += "The " + str(key) + " is: " + str(value) + "\n"
        return result

    def get_id_link_price(self, games: dict, name: str) -> dict:
        """The response may contain more than one game, so we need to choose the right one and
        return the id."""

        game_info = {}
        for app in games["apps"]:
            game_info["id"] = app["id"]
            game_info["link"] = app["link"]
            game_info["price"] = app["price"]
            break
        return game_info

    def remove_html_tags(html_string):
        soup = BeautifulSoup(html_string, "html.parser")
        return soup.get_text()

    def details_of_games(self, name: str) -> str:
        games = self.steam.apps.search_games(name)
        info_partOne_dict = self.get_id_link_price(games, name)
        info_partOne = self.parse_to_str(info_partOne_dict)
        id = str(info_partOne_dict.get("id"))
        info_dict = self.steam.apps.get_app_details(id)
        detailed_description = info_dict.get(id).get("data").get("detailed_description")

        # detailed_description contains <li> <br> some other html tags, so we need to remove them
        detailed_description = self.remove_html_tags(detailed_description)
        supported_languages = info_dict.get(id).get("data").get("supported_languages")
        info_partTwo = (
            "The detailed description of the game is: "
            + detailed_description
            + "\n"
            + "The supported languages of the game are: "
            + supported_languages
            + "\n"
        )
        info = info_partOne + info_partTwo

        return info

    def get_steam_id(self, name: str) -> str:
        user = self.steam.users.search_user(name)
        steam_id = user["player"]["steamid"]
        return steam_id

    def get_users_games(self, steam_id: str) -> list[str]:
        return self.steam.users.get_owned_games(steam_id, False, False)

    def recommended_games(self, steam_id: str) -> dict:
        users_games = self.get_users_games(steam_id)
        result = {}
        most_popular_genre = ""
        most_popular_genre_count = 0
        for game in users_games["games"]:
            appid = game["appid"]
            data_request = {"request": "appdetails", "appid": appid}
            genreStore = self.steamspypi.download(data_request)
            genreList = genreStore.get("genre", "").split(", ")
        
            for genre in genreList:
                if genre in result:
                    result[genre] += 1
                else:
                    result[genre] = 1
                if result[genre] > most_popular_genre_count:
                    most_popular_genre_count = result[genre]
                    most_popular_genre = genre

        data_request = dict()
        data_request['request'] = 'genre'
        data_request['genre'] = most_popular_genre
        data = steamspypi.download(data_request)
        sorted_data = sorted(data, key=lambda x: x.get('average_forever', 0), reverse=True)

        # Filter out games that the user already owns
        owned_games = [game["appid"] for game in users_games["games"]]
        remaining_games = [game for game in sorted_data if game["appid"] not in owned_games]

        # Select top 5 games by popularity from the remaining games list
        top_5_popular_not_owned = [game["name"] for game in remaining_games[:5]]
        # print(top_5_popular_not_owned)
        # Return the top 5 games by popularity that the user doesn't own
        return top_5_popular_not_owned
    


    def run(self, mode: str, game: str) -> str:
        if mode == "get_game_details":
            return self.details_of_games(game)
        elif mode == "get_recommended_games":
            return self.recommended_games(game)
        else:
            raise ValueError(f"Invalid mode {mode} for Steam API.")
