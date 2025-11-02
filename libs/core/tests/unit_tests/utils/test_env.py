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


def test_get_from_dict_or_env_empty_string() -> None:
    """Test that empty strings are returned consistently for list and string keys."""
    data = {"key": "", "other": "value"}

    # Test single string key with empty string value
    assert get_from_dict_or_env(data, "key", "ENV_KEY", default="default") == ""

    # Test list of keys with empty string value (should behave the same)
    assert get_from_dict_or_env(data, ["key"], "ENV_KEY", default="default") == ""

    # Test list of keys where first key has empty string
    result = get_from_dict_or_env(data, ["key", "other"], "ENV_KEY", default="default")
    assert result == ""


def test_get_from_dict_or_env_none_value() -> None:
    """Test that None values are skipped and default is used."""
    data = {"key": None, "other": "value"}

    # Test single string key with None value - should use default
    result = get_from_dict_or_env(data, "key", "ENV_KEY", default="default")
    assert result == "default"

    # Test list of keys with None value - should skip to next key
    result = get_from_dict_or_env(data, ["key", "other"], "ENV_KEY", default="default")
    assert result == "value"
