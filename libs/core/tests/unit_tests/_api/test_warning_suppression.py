import asyncio
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from contextvars import copy_context
from functools import partial
from typing import NamedTuple, cast

import pytest

from langchain_core._api.beta_decorator import (
    LangChainBetaWarning,
    beta,
    suppress_langchain_beta_warning,
    warn_beta,
)
from langchain_core._api.deprecation import (
    LangChainDeprecationWarning,
    LangChainPendingDeprecationWarning,
    deprecated,
    suppress_langchain_deprecation_warning,
    warn_deprecated,
)


class WarningAPI(NamedTuple):
    suppress: Callable[[], AbstractContextManager[None]]
    emit: Callable[[], None]
    category: type[Warning]


@pytest.fixture(
    params=[
        WarningAPI(
            suppress_langchain_deprecation_warning,
            partial(warn_deprecated, "1.0"),
            LangChainDeprecationWarning,
        ),
        WarningAPI(
            suppress_langchain_deprecation_warning,
            partial(warn_deprecated, "1.0", pending=True),
            LangChainPendingDeprecationWarning,
        ),
        WarningAPI(suppress_langchain_beta_warning, warn_beta, LangChainBetaWarning),
    ],
    ids=["deprecation", "pending", "beta"],
)
def warning_api(request: pytest.FixtureRequest) -> WarningAPI:
    return cast("WarningAPI", request.param)


def test_suppression_does_not_change_warning_filters(warning_api: WarningAPI) -> None:
    """Only warnings emitted through the helpers are suppressed."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", warning_api.category)
        filters = list(warnings.filters)
        with warning_api.suppress():
            warning_api.emit()
            assert warnings.filters == filters
            with pytest.raises(warning_api.category, match="external"):
                warnings.warn("external", warning_api.category, stacklevel=1)


@pytest.mark.parametrize("direct", [False, True], ids=["helper", "direct"])
def test_suppression_is_thread_local(warning_api: WarningAPI, *, direct: bool) -> None:
    """An unrelated thread retains warnings-as-errors during suppression."""
    emit = (
        partial(warnings.warn, "external", warning_api.category, stacklevel=1)
        if direct
        else warning_api.emit
    )
    with warnings.catch_warnings(), ThreadPoolExecutor(max_workers=1) as executor:
        warnings.simplefilter("error", warning_api.category)
        context = copy_context()
        with warning_api.suppress():
            warning_api.emit()
            with pytest.raises(warning_api.category):
                executor.submit(context.run, emit).result(timeout=5)


def test_nested_suppression_restores_after_exception(warning_api: WarningAPI) -> None:
    """Nested contexts restore suppression even when either scope raises."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", warning_api.category)
        with warning_api.suppress():
            warning_api.emit()
            with pytest.raises(ZeroDivisionError), warning_api.suppress():
                raise ZeroDivisionError
            warning_api.emit()
        with pytest.raises(ZeroDivisionError), warning_api.suppress():
            raise ZeroDivisionError
        with pytest.raises(warning_api.category):
            warning_api.emit()


async def test_suppression_follows_context(warning_api: WarningAPI) -> None:
    """Tasks and propagated thread contexts inherit only their own suppression."""

    async def emit_when_ready(ready: asyncio.Event) -> None:
        await ready.wait()
        warning_api.emit()

    with warnings.catch_warnings():
        warnings.simplefilter("error", warning_api.category)
        unrelated_ready = asyncio.Event()
        inherited_ready = asyncio.Event()
        unrelated = asyncio.create_task(emit_when_ready(unrelated_ready))
        with warning_api.suppress():
            unrelated_ready.set()
            with pytest.raises(warning_api.category):
                await unrelated
            inherited = asyncio.create_task(emit_when_ready(inherited_ready))
            await asyncio.to_thread(warning_api.emit)
        inherited_ready.set()
        await inherited
        with pytest.raises(warning_api.category):
            warning_api.emit()


async def test_suppressed_decorators_preserve_warning_state(
    warning_api: WarningAPI,
) -> None:
    """Suppression preserves each decorator's once-state and property warnings."""
    is_beta = warning_api.category is LangChainBetaWarning
    decorator = (
        beta()
        if is_beta
        else deprecated(
            "1.0",
            package="tests",
            pending=warning_api.category is LangChainPendingDeprecationWarning,
        )
    )

    @decorator
    def function() -> str:
        return "value"

    @decorator
    async def async_function() -> str:
        return "value"

    @decorator
    class AnnotatedClass:
        def __init__(self) -> None:
            pass

    class WithProperty:
        def get_value(self) -> str:
            return "value"

        def set_value(self, _value: str) -> None:
            pass

        def delete_value(self) -> None:
            pass

        value = decorator(property(get_value, set_value, delete_value))

    obj = WithProperty()
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        with warning_api.suppress():
            assert function() == "value"
            assert await async_function() == "value"
            AnnotatedClass()
            assert obj.value == "value"
            obj.value = "value"
            del obj.value
        assert not recorded

        assert function() == "value"
        assert await async_function() == "value"
        AnnotatedClass()
        assert obj.value == "value"
        obj.value = "value"
        del obj.value
        assert len(recorded) == 6
        assert all(w.category is warning_api.category for w in recorded)
        assert all(w.filename == __file__ for w in recorded)

        assert function() == "value"
        assert await async_function() == "value"
        AnnotatedClass()
        assert len(recorded) == 6
