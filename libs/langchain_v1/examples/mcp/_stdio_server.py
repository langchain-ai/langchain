"""The weather server as a standalone script, for the stdio example."""

from _servers import weather_server

if __name__ == "__main__":
    weather_server().run()
