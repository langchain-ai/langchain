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


def test_env_var_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test checking environment variable truthiness."""
    from langchain_core.utils.env import env_var_is_set

    # Unset variable
    monkeypatch.delenv("TEST_LANGCHAIN_FLAG", raising=False)
    assert not env_var_is_set("TEST_LANGCHAIN_FLAG")

    # Set to true-like values
    monkeypatch.setenv("TEST_LANGCHAIN_FLAG", "1")
    assert env_var_is_set("TEST_LANGCHAIN_FLAG")

    monkeypatch.setenv("TEST_LANGCHAIN_FLAG", "true")
    assert env_var_is_set("TEST_LANGCHAIN_FLAG")

    # Set to false-like values
    monkeypatch.setenv("TEST_LANGCHAIN_FLAG", "0")
    assert not env_var_is_set("TEST_LANGCHAIN_FLAG")

    monkeypatch.setenv("TEST_LANGCHAIN_FLAG", "false")
    assert not env_var_is_set("TEST_LANGCHAIN_FLAG")

    monkeypatch.setenv("TEST_LANGCHAIN_FLAG", "")
    assert not env_var_is_set("TEST_LANGCHAIN_FLAG")


def test_get_from_dict_or_env_with_tuple_keys() -> None:
    """Test get_from_dict_or_env with tuple of fallback keys."""
    data = {"secondary_key": "bar"}
    assert (
        get_from_dict_or_env(
            data,
            ("primary_key", "secondary_key"),
            "__ENV_KEY",
        )
        == "bar"
    )
