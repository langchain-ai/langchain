import ast

from langchain_community.utilities.steam import SteamWebAPIWrapper


def test_get_game_details() -> None:
    """Test for getting game details on Steam"""
    steam = SteamWebAPIWrapper()  # type: ignore[call-arg]
    output = steam.run("get_game_details", "Terraria")
    assert "id" in output
    assert "link" in output
    assert "detailed description" in output
    assert "supported languages" in output
    assert "price" in output


def test_get_recommended_games() -> None:
    """Test for getting recommended games on Steam"""
    steam = SteamWebAPIWrapper()  # type: ignore[call-arg]
    output = steam.run("get_recommended_games", "76561198362745711")
    output = ast.literal_eval(output)
    assert len(output) == 5
