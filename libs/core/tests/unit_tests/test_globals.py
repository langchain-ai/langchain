from langchain_core.globals import get_debug, set_debug


def test_debug_is_settable_via_setter() -> None:
    from langchain_core import globals as globals_
    from langchain_core.callbacks.manager import _get_debug

    previous_value = globals_._debug
    previous_fn_reading = _get_debug()
    assert previous_value == previous_fn_reading

    # Flip the value of the flag.
    set_debug(not previous_value)

    new_value = globals_._debug
    new_fn_reading = _get_debug()

    try:
        # We successfully changed the value of `debug`.
        assert new_value != previous_value

        # If we access `debug` via a function used elsewhere in langchain,
        # it also sees the same new value.
        assert new_value == new_fn_reading

        # If we access `debug` via `get_debug()` we also get the same value.
        assert new_value == get_debug()
    finally:
        # Make sure we don't alter global state, even if the test fails.
        # Always reset `debug` to the value it had before.
        set_debug(previous_value)
