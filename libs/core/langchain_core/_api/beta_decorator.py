"""Helper functions for marking parts of the LangChain API as beta.

This module was loosely adapted from matplotlibs _api/deprecation.py module:

https://github.com/matplotlib/matplotlib/blob/main/lib/matplotlib/_api/deprecation.py

.. warning::

    This module is for internal use only.  Do not use it in your own code.
    We may change the API at any time with no warning.
"""

import contextlib
import functools
import inspect
import warnings
from collections.abc import Generator
from typing import Any, Callable, TypeVar, Union, cast

from langchain_core._api.internal import is_caller_internal


class LangChainBetaWarning(DeprecationWarning):
    """A class for issuing beta warnings for LangChain users."""


# PUBLIC API


T = TypeVar("T", bound=Union[Callable[..., Any], type])


def beta(
    *,
    message: str = "",
    name: str = "",
    obj_type: str = "",
    addendum: str = "",
) -> Callable[[T], T]:
    """Decorator to mark a function, a class, or a property as beta.

    When marking a classmethod, a staticmethod, or a property, the
    ``@beta`` decorator should go *under* ``@classmethod`` and
    ``@staticmethod`` (i.e., `beta` should directly decorate the
    underlying callable), but *over* ``@property``.

    When marking a class ``C`` intended to be used as a base class in a
    multiple inheritance hierarchy, ``C`` *must* define an ``__init__`` method
    (if ``C`` instead inherited its ``__init__`` from its own base class, then
    ``@beta`` would mess up ``__init__`` inheritance when installing its
    own (annotation-emitting) ``C.__init__``).

    Args:
        message : str, optional
            Override the default beta message. The %(since)s,
            %(name)s, %(alternative)s, %(obj_type)s, %(addendum)s,
            and %(removal)s format specifiers will be replaced by the
            values of the respective arguments passed to this function.
        name : str, optional
            The name of the beta object.
        obj_type : str, optional
            The object type being beta.
        addendum : str, optional
            Additional text appended directly to the final message.

    Examples:

        .. code-block:: python

            @beta
            def the_function_to_annotate():
                pass

    """

    def beta(
        obj: T,
        *,
        _obj_type: str = obj_type,
        _name: str = name,
        _message: str = message,
        _addendum: str = addendum,
    ) -> T:
        """Implementation of the decorator returned by `beta`."""

        def emit_warning() -> None:
            """Emit the warning."""
            warn_beta(
                message=_message,
                name=_name,
                obj_type=_obj_type,
                addendum=_addendum,
            )

        warned = False

        def warning_emitting_wrapper(*args: Any, **kwargs: Any) -> Any:
            """Wrapper for the original wrapped callable that emits a warning.

            Args:
                *args: The positional arguments to the function.
                **kwargs: The keyword arguments to the function.

            Returns:
                The return value of the function being wrapped.
            """
            nonlocal warned
            if not warned and not is_caller_internal():
                warned = True
                emit_warning()
            return wrapped(*args, **kwargs)

        async def awarning_emitting_wrapper(*args: Any, **kwargs: Any) -> Any:
            """Same as warning_emitting_wrapper, but for async functions."""
            nonlocal warned
            if not warned and not is_caller_internal():
                warned = True
                emit_warning()
            return await wrapped(*args, **kwargs)

        if isinstance(obj, type):
            if not _obj_type:
                _obj_type = "class"
            wrapped = obj.__init__  # type: ignore[misc]
            _name = _name or obj.__qualname__
            old_doc = obj.__doc__

            def finalize(wrapper: Callable[..., Any], new_doc: str) -> T:  # noqa: ARG001
                """Finalize the annotation of a class."""
                # Can't set new_doc on some extension objects.
                with contextlib.suppress(AttributeError):
                    obj.__doc__ = new_doc

                def warn_if_direct_instance(
                    self: Any, *args: Any, **kwargs: Any
                ) -> Any:
                    """Warn that the class is in beta."""
                    nonlocal warned
                    if not warned and type(self) is obj and not is_caller_internal():
                        warned = True
                        emit_warning()
                    return wrapped(self, *args, **kwargs)

                obj.__init__ = functools.wraps(obj.__init__)(  # type: ignore[misc]
                    warn_if_direct_instance
                )
                return obj

        elif isinstance(obj, property):
            if not _obj_type:
                _obj_type = "attribute"
            wrapped = None
            _name = _name or obj.fget.__qualname__
            old_doc = obj.__doc__

            class _BetaProperty(property):
                """A beta property."""

                def __init__(
                    self,
                    fget: Union[Callable[[Any], Any], None] = None,
                    fset: Union[Callable[[Any, Any], None], None] = None,
                    fdel: Union[Callable[[Any], None], None] = None,
                    doc: Union[str, None] = None,
                ) -> None:
                    super().__init__(fget, fset, fdel, doc)
                    self.__orig_fget = fget
                    self.__orig_fset = fset
                    self.__orig_fdel = fdel
                    self.__doc__ = doc

                def __get__(
                    self, instance: Any, owner: Union[type, None] = None
                ) -> Any:
                    if instance is not None or owner is not None:
                        emit_warning()
                    return self.fget(instance)

                def __set__(self, instance: Any, value: Any) -> None:
                    if instance is not None:
                        emit_warning()
                    return self.fset(instance, value)

                def __delete__(self, instance: Any) -> None:
                    if instance is not None:
                        emit_warning()
                    return self.fdel(instance)

                def __set_name__(self, owner: Union[type, None], set_name: str) -> None:
                    nonlocal _name
                    if _name == "<lambda>":
                        _name = set_name

            def finalize(wrapper: Callable[..., Any], new_doc: str) -> Any:  # noqa: ARG001
                """Finalize the property."""
                return _BetaProperty(
                    fget=obj.fget, fset=obj.fset, fdel=obj.fdel, doc=new_doc
                )

        else:
            _name = _name or obj.__qualname__
            if not _obj_type:
                # edge case: when a function is within another function
                # within a test, this will call it a "method" not a "function"
                _obj_type = "function" if "." not in _name else "method"
            wrapped = obj
            old_doc = wrapped.__doc__

            def finalize(wrapper: Callable[..., Any], new_doc: str) -> T:
                """Wrap the wrapped function using the wrapper and update the docstring.

                Args:
                    wrapper: The wrapper function.
                    new_doc: The new docstring.

                Returns:
                    The wrapped function.
                """
                wrapper = functools.wraps(wrapped)(wrapper)
                wrapper.__doc__ = new_doc
                return cast("T", wrapper)

        old_doc = inspect.cleandoc(old_doc or "").strip("\n") or ""
        components = [message, addendum]
        details = " ".join([component.strip() for component in components if component])
        new_doc = f".. beta::\n   {details}\n\n{old_doc}\n"

        if inspect.iscoroutinefunction(obj):
            return finalize(awarning_emitting_wrapper, new_doc)
        return finalize(warning_emitting_wrapper, new_doc)

    return beta


@contextlib.contextmanager
def suppress_langchain_beta_warning() -> Generator[None, None, None]:
    """Context manager to suppress LangChainDeprecationWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LangChainBetaWarning)
        yield


def warn_beta(
    *,
    message: str = "",
    name: str = "",
    obj_type: str = "",
    addendum: str = "",
) -> None:
    """Display a standardized beta annotation.

    Arguments:
        message : str, optional
            Override the default beta message. The
            %(name)s, %(obj_type)s, %(addendum)s
            format specifiers will be replaced by the
            values of the respective arguments passed to this function.
        name : str, optional
            The name of the annotated object.
        obj_type : str, optional
            The object type being annotated.
        addendum : str, optional
            Additional text appended directly to the final message.
    """
    if not message:
        message = ""

        if obj_type:
            message += f"The {obj_type} `{name}`"
        else:
            message += f"`{name}`"

        message += " is in beta. It is actively being worked on, so the API may change."

        if addendum:
            message += f" {addendum}"

    warning = LangChainBetaWarning(message)
    warnings.warn(warning, category=LangChainBetaWarning, stacklevel=4)


def surface_langchain_beta_warnings() -> None:
    """Unmute LangChain beta warnings."""
    warnings.filterwarnings(
        "default",
        category=LangChainBetaWarning,
    )
