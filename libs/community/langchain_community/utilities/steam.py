"""Util that calls Steam-WebAPI."""

from typing import Any, List

from pydantic import BaseModel, ConfigDict, model_validator

from langchain_community.tools.steam.prompt import (
    STEAM_GET_GAMES_DETAILS,
    STEAM_GET_RECOMMENDED_GAMES,
)


class SteamWebAPIWrapper(BaseModel):
    """Wrapper for Steam API."""

    steam: Any = None  # for python-steam-api

    # operations: a list of dictionaries, each representing a specific operation that
    # can be performed with the API
    operations: List[dict] = [
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

    model_config = ConfigDict(
        extra="forbid",
    )

    def get_operations(self) -> List[dict]:
        """Return a list of operations."""
        return self.operations

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict) -> Any:
        """Validate api key and python package has been configured."""

        # check if the python package is installed
        try:
            from steam import Steam
        except ImportError:
            raise ImportError("python-steam-api library is not installed. ")

        try:
            from decouple import config
        except ImportError:
            raise ImportError("decouple library is not installed. ")

        # initialize the steam attribute for python-steam-api usage
        KEY = config("STEAM_KEY")
        steam = Steam(KEY)
        values["steam"] = steam
        return values

    def parse_to_str(self, details: dict) -> str:  # For later parsing
        """Parse the details result."""
        result = ""
        for key, value in details.items():
            result += "The " + str(key) + " is: " + str(value) + "\n"
        return result

    def get_id_link_price(self, games: dict) -> dict:
        """The response may contain more than one game, so we need to choose the right
        one and return the id."""

        game_info = {}
        for app in games["apps"]:
            game_info["id"] = app["id"]
            game_info["link"] = app["link"]
            game_info["price"] = app["price"]
            break
        return game_info

    def remove_html_tags(self, html_string: str) -> str:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_string, "html.parser")
        return soup.get_text()

    def details_of_games(self, name: str) -> str:
        games = self.steam.apps.search_games(name)
        info_partOne_dict = self.get_id_link_price(games)
        info_partOne = self.parse_to_str(info_partOne_dict)
        id = str(info_partOne_dict.get("id"))
        info_dict = self.steam.apps.get_app_details(id)
        data = info_dict.get(id).get("data")
        detailed_description = data.get("detailed_description")

        # detailed_description contains <li> <br> some other html tags, so we need to
        # remove them
        detailed_description = self.remove_html_tags(detailed_description)
        supported_languages = info_dict.get(id).get("data").get("supported_languages")
        info_partTwo = (
            "The summary of the game is: "
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

    def get_users_games(self, steam_id: str) -> List[str]:
        return self.steam.users.get_owned_games(steam_id, False, False)

    def recommended_games(self, steam_id: str) -> str:
        try:
            import steamspypi
        except ImportError:
            raise ImportError("steamspypi library is not installed.")
        users_games = self.get_users_games(steam_id)
        result: dict[str, int] = {}
        most_popular_genre = ""
        most_popular_genre_count = 0
        for game in users_games["games"]:  # type: ignore[call-overload]
            appid = game["appid"]
            data_request = {"request": "appdetails", "appid": appid}
            genreStore = steamspypi.download(data_request)
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
        data_request["request"] = "genre"
        data_request["genre"] = most_popular_genre
        data = steamspypi.download(data_request)
        sorted_data = sorted(
            data.values(), key=lambda x: x.get("average_forever", 0), reverse=True
        )
        owned_games = [game["appid"] for game in users_games["games"]]  # type: ignore[call-overload]
        remaining_games = [
            game for game in sorted_data if game["appid"] not in owned_games
        ]
        top_5_popular_not_owned = [game["name"] for game in remaining_games[:5]]
        return str(top_5_popular_not_owned)

    def run(self, mode: str, game: str) -> str:
        if mode == "get_games_details":
            return self.details_of_games(game)
        elif mode == "get_recommended_games":
            return self.recommended_games(game)
        else:
            raise ValueError(f"Invalid mode {mode} for Steam API.")
