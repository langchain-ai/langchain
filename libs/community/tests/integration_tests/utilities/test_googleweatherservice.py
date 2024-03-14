from libs.community.langchain_community.utilities.google_weatherservice import (
    GoogleNationalWeatherAPI,
)


def test_googlenationalweatherapi() -> None:
    """Test that the GoogleNationalWeatherAPI returns correct data for Chicago, IL"""

    weather = GoogleNationalWeatherAPI()
    weather_data = weather._run("Chicago")

    assert weather_data is not None
    assert "Chicago" in weather_data
    assert "Fahrenheit" in weather_data
