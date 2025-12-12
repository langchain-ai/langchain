"""Pytest conftest."""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest
import yaml
from langchain_core._api.deprecation import deprecated
from vcr import VCR
from vcr.persisters.filesystem import CassetteNotFoundError
from vcr.request import Request

if TYPE_CHECKING:
    from os import PathLike


class CustomSerializer:
    """Custom serializer for VCR cassettes using YAML and gzip.

    We're using a custom serializer to avoid the default yaml serializer
    used by VCR, which is not designed to be safe for untrusted input.

    This step is an extra precaution necessary because the cassette files
    are in compressed YAML format, which makes it more difficult to inspect
    their contents during development or debugging.
    """

    @staticmethod
    def serialize(cassette_dict: dict[str, Any]) -> bytes:
        """Convert cassette to YAML and compress it."""
        cassette_dict["requests"] = [
            {
                "method": request.method,
                "uri": request.uri,
                "body": request.body,
                "headers": {k: [v] for k, v in request.headers.items()},
            }
            for request in cassette_dict["requests"]
        ]
        yml = yaml.safe_dump(cassette_dict)
        return gzip.compress(yml.encode("utf-8"))

    @staticmethod
    def deserialize(data: bytes) -> dict[str, Any]:
        """Decompress data and convert it from YAML."""
        decoded_yaml = gzip.decompress(data).decode("utf-8")
        cassette = cast("dict[str, Any]", yaml.safe_load(decoded_yaml))
        cassette["requests"] = [Request(**request) for request in cassette["requests"]]
        return cassette


class CustomPersister:
    """A custom persister for VCR that uses the `CustomSerializer`."""

    @classmethod
    def load_cassette(
        cls,
        cassette_path: str | PathLike[str],
        serializer: CustomSerializer,
    ) -> tuple[list[Any], list[Any]]:
        """Load a cassette from a file."""
        # If cassette path is already Path this is a no-op
        cassette_path = Path(cassette_path)
        if not cassette_path.is_file():
            msg = f"Cassette file {cassette_path} does not exist."
            raise CassetteNotFoundError(msg)
        with cassette_path.open(mode="rb") as f:
            data = f.read()
        deser = serializer.deserialize(data)
        return deser["requests"], deser["responses"]

    @staticmethod
    def save_cassette(
        cassette_path: str | PathLike[str],
        cassette_dict: dict[str, Any],
        serializer: CustomSerializer,
    ) -> None:
        """Save a cassette to a file."""
        data = serializer.serialize(cassette_dict)
        # if cassette path is already Path this is no operation
        cassette_path = Path(cassette_path)
        cassette_folder = cassette_path.parent
        if not cassette_folder.exists():
            cassette_folder.mkdir(parents=True)
        with cassette_path.open("wb") as f:
            f.write(data)


# A list of headers that should be filtered out of the cassettes.
# These are typically associated with sensitive information and should
# not be stored in cassettes.
_BASE_FILTER_HEADERS = [
    ("authorization", "PLACEHOLDER"),
    ("x-api-key", "PLACEHOLDER"),
    ("api-key", "PLACEHOLDER"),
]


def base_vcr_config() -> dict[str, Any]:
    """Return VCR configuration that every cassette will receive.

    (Anything permitted by `vcr.VCR(**kwargs)` can be put here.)
    """
    return {
        "record_mode": "once",
        "filter_headers": _BASE_FILTER_HEADERS.copy(),
        "match_on": ["method", "uri", "body"],
        "allow_playback_repeats": True,
        "decode_compressed_response": True,
        "cassette_library_dir": "tests/cassettes",
        "path_transformer": VCR.ensure_suffix(".yaml"),
    }


@pytest.fixture(scope="session")
@deprecated("1.0.3", alternative="base_vcr_config", removal="2.0")
def _base_vcr_config() -> dict[str, Any]:
    return base_vcr_config()


@pytest.fixture(scope="session")
def vcr_config() -> dict[str, Any]:
    """VCR config fixture."""
    return base_vcr_config()
