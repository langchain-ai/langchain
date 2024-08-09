import os
from collections.abc import Generator
from contextlib import contextmanager

API_KEY_PIECES = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJuZXhhOmE5YTk2OTRlLTMwZDEtN",
    "zA0NC1lNGNlLWU1N2M4YWRjNWYyNDpnb29nbGU6YmFzaWMiLCJpYXQiOjE3MTk5NTAwMDAsImV",
    "4cCI6MTc1MTQ4NjAwMH0.ro2QjTW4sZMaC43QtxuaUJbDRjgBAPnomoruv92KdDY",
)


@contextmanager
def temporary_api_key() -> Generator:
    if "NEXA_API_KEY" not in os.environ:
        # this is a temporary key only for testing purposes
        os.environ["NEXA_API_KEY"] = "".join(API_KEY_PIECES)
        try:
            yield
        finally:
            del os.environ["NEXA_API_KEY"]
    else:
        yield
