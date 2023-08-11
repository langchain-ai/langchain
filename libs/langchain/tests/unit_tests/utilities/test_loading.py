"""Test the functionality of loading from langchain-hub."""

import json
import re
from pathlib import Path
from typing import Iterable
from unittest.mock import Mock
from urllib.parse import urljoin

import pytest
import responses

from langchain.utilities.loading import DEFAULT_REF, URL_BASE, try_load_from_hub


@pytest.fixture(autouse=True)
def mocked_responses() -> Iterable[responses.RequestsMock]:
    """Fixture mocking requests.get."""
    with responses.RequestsMock() as rsps:
        yield rsps


def test_non_hub_path() -> None:
    """Test that a non-hub path returns None."""
    path = "chains/some_path"
    loader = Mock()
    valid_suffixes = {"suffix"}
    result = try_load_from_hub(path, loader, "chains", valid_suffixes)

    assert result is None
    loader.assert_not_called()


def test_invalid_prefix() -> None:
    """Test that a hub path with an invalid prefix returns None."""
    path = "lc://agents/some_path"
    loader = Mock()
    valid_suffixes = {"suffix"}
    result = try_load_from_hub(path, loader, "chains", valid_suffixes)

    assert result is None
    loader.assert_not_called()


def test_invalid_suffix() -> None:
    """Test that a hub path with an invalid suffix raises an error."""
    path = "lc://chains/path.invalid"
    loader = Mock()
    valid_suffixes = {"json"}

    with pytest.raises(ValueError, match="Unsupported file type."):
        try_load_from_hub(path, loader, "chains", valid_suffixes)

    loader.assert_not_called()


@pytest.mark.parametrize("ref", [None, "v0.3"])
def test_success(mocked_responses: responses.RequestsMock, ref: str) -> None:
    """Test that a valid hub path is loaded correctly with and without a ref."""
    path = "chains/path/chain.json"
    lc_path_prefix = f"lc{('@' + ref) if ref else ''}://"
    valid_suffixes = {"json"}
    body = json.dumps({"foo": "bar"})
    ref = ref or DEFAULT_REF

    file_contents = None

    def loader(file_path: str) -> None:
        nonlocal file_contents
        assert file_contents is None
        file_contents = Path(file_path).read_text()

    mocked_responses.get(  # type: ignore
        urljoin(URL_BASE.format(ref=ref), path),
        body=body,
        status=200,
        content_type="application/json",
    )

    try_load_from_hub(f"{lc_path_prefix}{path}", loader, "chains", valid_suffixes)
    assert file_contents == body


def test_failed_request(mocked_responses: responses.RequestsMock) -> None:
    """Test that a failed request raises an error."""
    path = "chains/path/chain.json"
    loader = Mock()

    mocked_responses.get(  # type: ignore
        urljoin(URL_BASE.format(ref=DEFAULT_REF), path), status=500
    )

    with pytest.raises(ValueError, match=re.compile("Could not find file at .*")):
        try_load_from_hub(f"lc://{path}", loader, "chains", {"json"})
    loader.assert_not_called()
