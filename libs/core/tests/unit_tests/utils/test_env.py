import pytest

from langchain_core.utils.env import get_from_dict_or_env


class _MaskedValue(str):
    __slots__ = ()

    def __str__(self) -> str:
        return "masked"


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


def test_get_from_dict_or_env_does_not_coerce_with_str() -> None:
    value = _MaskedValue("original")
    result = get_from_dict_or_env(
        {
            "a": value,
        },
        "a",
        "__SOME_KEY_IN_ENV",
    )

    assert isinstance(result, _MaskedValue)
    assert result == "original"
