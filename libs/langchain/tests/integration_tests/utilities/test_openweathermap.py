from langchain.utilities.openweathermap import OpenWeatherMapAPIWrapper


def test_openweathermap_api_wrapper() -> None:
    """Test that OpenWeatherMapAPIWrapper returns correct data for London, GB."""

    weather = OpenWeatherMapAPIWrapper()
    weather_data = weather.run("London,GB")

    assert weather_data is not None
    assert "London" in weather_data
    assert "GB" in weather_data
    assert "Detailed status:" in weather_data
    assert "Wind speed:" in weather_data
    assert "direction:" in weather_data
    assert "Humidity:" in weather_data
    assert "Temperature:" in weather_data
    assert "Current:" in weather_data
    assert "High:" in weather_data
    assert "Low:" in weather_data
    assert "Feels like:" in weather_data
    assert "Rain:" in weather_data
    assert "Heat index:" in weather_data
    assert "Cloud cover:" in weather_data
