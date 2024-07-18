STEAM_GET_GAMES_DETAILS = """
    This tool is a wrapper around python-steam-api's steam.apps.search_games API and 
    steam.apps.get_app_details API, useful when you need to search for a game.
    The input to this tool is a string specifying the name of the game you want to 
    search for. For example, to search for a game called "Counter-Strike: Global 
    Offensive", you would input "Counter-Strike: Global Offensive" as the game name.
    This input will be passed into steam.apps.search_games to find the game id, link 
    and price, and then the game id will be passed into steam.apps.get_app_details to 
    get the detailed description and supported languages of the game. Finally the 
    results are combined and returned as a string.
"""

STEAM_GET_RECOMMENDED_GAMES = """
    This tool is a wrapper around python-steam-api's steam.users.get_owned_games API 
    and steamspypi's steamspypi.download API, useful when you need to get a list of 
    recommended games. The input to this tool is a string specifying the steam id of 
    the user you want to get recommended games for. For example, to get recommended 
    games for a user with steam id 76561197960435530, you would input 
    "76561197960435530" as the steam id.  This steamid is then utilized to form a 
    data_request sent to steamspypi's steamspypi.download to retrieve genres of user's 
    owned games. Then, calculates the frequency of each genre, identifying the most 
    popular one, and stored it in a dictionary. Subsequently, use steamspypi.download
    to returns all games in this genre and return 5 most-played games that is not owned
    by the user.

"""
