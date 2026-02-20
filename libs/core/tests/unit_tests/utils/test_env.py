from enum import Enum

import pytest

from langchain_core.utils.env import get_from_dict_or_env


def test_get_from_dict_or_env() -> None:
    assert (
        get_from_dict_or_env(
            {
                "a": "foo",
            },
            ["a"],
            "__SOME_KEY_IN_ENV",
        )
        == "foo"
    )

    assert (
        get_from_dict_or_env(
            {
                "a": "foo",
            },
            ["b", "a"],
            "__SOME_KEY_IN_ENV",
        )
        == "foo"
    )

    assert (
        get_from_dict_or_env(
            {
                "a": "foo",
            },
            "a",
            "__SOME_KEY_IN_ENV",
        )
        == "foo"
    )

    assert (
        get_from_dict_or_env(
            {
                "a": "foo",
            },
            "not exists",
            "__SOME_KEY_IN_ENV",
            default="default",
        )
        == "default"
    )

    # Not the most obvious behavior, but
    # this is how it works right now
    with pytest.raises(
        ValueError,
        match="Did not find not exists, "
        "please add an environment variable `__SOME_KEY_IN_ENV` which contains it, "
        "or pass `not exists` as a named parameter",
    ):
        assert (
            get_from_dict_or_env(
                {
                    "a": "foo",
                },
                "not exists",
                "__SOME_KEY_IN_ENV",
            )
            is None
        )


def test_get_from_dict_or_env_preserves_non_string_types() -> None:
    """Non-string dict values should be returned as-is without str() cast."""

    class Status(Enum):
        ACTIVE = "active"
        INACTIVE = "inactive"

    # Enum value with string key
    result = get_from_dict_or_env(
        {"status": Status.ACTIVE},
        "status",
        "__SOME_KEY_IN_ENV",
    )
    assert result is Status.ACTIVE

    # Enum value with list key
    result = get_from_dict_or_env(
        {"status": Status.INACTIVE},
        ["status"],
        "__SOME_KEY_IN_ENV",
    )
    assert result is Status.INACTIVE

    # Integer value
    result = get_from_dict_or_env(
        {"port": 8080},
        "port",
        "__SOME_KEY_IN_ENV",
    )
    assert result == 8080
    assert isinstance(result, int)


def test_get_from_dict_or_env_falls_back_to_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When key is not in dict, should fall back to environment variable."""
    monkeypatch.setenv("__TEST_ENV_KEY", "from_env")
    result = get_from_dict_or_env({}, "missing", "__TEST_ENV_KEY")
    assert result == "from_env"
    assert isinstance(result, str)
