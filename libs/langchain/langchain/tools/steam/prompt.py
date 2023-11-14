STEAM_GET_GAMES_ID = """
    This tool is a wrapper around python-steam-api's Searching for Games API(steam.apps.search_games), useful when you want to get the ID of a specific game on Steam.
    The input to this tool is a string specifying the name of the game you want to search for, and will be passed into python-steam-api's 'steam.apps.search_games' function.
    For example, to search for a game called "terr", you would input "terr" as the name of the game.

"""

STEAM_GET_GAMES_DETAILS = """
    This tool is a wrapper around steamspypi's Returns details for a given application API(steamspypi.download), useful when you want to get details of the game.
    The input to this tool is a dictionary specifying the request and appid of the game you want to search for, and will be passed into steamspypi's 'steamspypi.download' function.
    For example, to search for a game with id "730", you would input {'request': 'appdetails', 'appid': '730'} as the dictionary.

"""

STEAM_GET_RECOMMENDED_GAMES = """
    This tool is a wrapper around python-steam-api's Searching for a user 'steam.users.search_user' and Getting owned games 'steam.users.get_owned_games' API, useful for when you 
    want to get recommended games for a user. The input to this tool is a string specifying the username of the user you want recommended games for. For example, to search for 
    recommended games for a user called "John", you would input "John" as the username.

"""

# need more discussion on more functionalities of steam api