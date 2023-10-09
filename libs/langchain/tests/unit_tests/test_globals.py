from langchain.globals import get_debug, get_verbose, set_debug, set_verbose


def test_debug_is_settable_directly() -> None:
    import langchain
    from langchain.callbacks.manager import _get_debug

    previous_value = langchain.debug
    previous_fn_reading = _get_debug()
    assert previous_value == previous_fn_reading

    # Flip the value of the flag.
    langchain.debug = not previous_value

    new_value = langchain.debug
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


def test_debug_is_settable_via_setter() -> None:
    from langchain import globals
    from langchain.callbacks.manager import _get_debug

    previous_value = globals._debug
    previous_fn_reading = _get_debug()
    assert previous_value == previous_fn_reading

    # Flip the value of the flag.
    set_debug(not previous_value)

    new_value = globals._debug
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


def test_verbose_is_settable_directly() -> None:
    import langchain
    from langchain.chains.base import _get_verbosity

    previous_value = langchain.verbose
    previous_fn_reading = _get_verbosity()
    assert previous_value == previous_fn_reading

    # Flip the value of the flag.
    langchain.verbose = not previous_value

    new_value = langchain.verbose
    new_fn_reading = _get_verbosity()

    try:
        # We successfully changed the value of `verbose`.
        assert new_value != previous_value

        # If we access `verbose` via a function used elsewhere in langchain,
        # it also sees the same new value.
        assert new_value == new_fn_reading

        # If we access `verbose` via `get_verbose()` we also get the same value.
        assert new_value == get_verbose()
    finally:
        # Make sure we don't alter global state, even if the test fails.
        # Always reset `verbose` to the value it had before.
        set_verbose(previous_value)


def test_verbose_is_settable_via_setter() -> None:
    from langchain import globals
    from langchain.chains.base import _get_verbosity

    previous_value = globals._verbose
    previous_fn_reading = _get_verbosity()
    assert previous_value == previous_fn_reading

    # Flip the value of the flag.
    set_verbose(not previous_value)

    new_value = globals._verbose
    new_fn_reading = _get_verbosity()

    try:
        # We successfully changed the value of `verbose`.
        assert new_value != previous_value

        # If we access `verbose` via a function used elsewhere in langchain,
        # it also sees the same new value.
        assert new_value == new_fn_reading

        # If we access `verbose` via `get_verbose()` we also get the same value.
        assert new_value == get_verbose()
    finally:
        # Make sure we don't alter global state, even if the test fails.
        # Always reset `verbose` to the value it had before.
        set_verbose(previous_value)
