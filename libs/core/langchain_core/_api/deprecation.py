"""Helper functions for deprecating parts of the LangChain API.

This module was adapted from matplotlibs _api/deprecation.py module:

https://github.com/matplotlib/matplotlib/blob/main/lib/matplotlib/_api/deprecation.py

.. warning::

    This module is for internal use only.  Do not use it in your own code.
    We may change the API at any time with no warning.
"""

import contextlib
import functools
import inspect
import warnings
from typing import Any, Callable, Generator, Type, TypeVar

from langchain_core._api.internal import is_caller_internal


class LangChainDeprecationWarning(DeprecationWarning):
    """A class for issuing deprecation warnings for LangChain users."""


class LangChainPendingDeprecationWarning(PendingDeprecationWarning):
    """A class for issuing deprecation warnings for LangChain users."""


# PUBLIC API


T = TypeVar("T", Type, Callable)


def deprecated(
    since: str,
    *,
    message: str = "",
    name: str = "",
    alternative: str = "",
    alternative_import: str = "",
    pending: bool = False,
    obj_type: str = "",
    addendum: str = "",
    removal: str = "",
) -> Callable[[T], T]:
    """Decorator to mark a function, a class, or a property as deprecated.

    When deprecating a classmethod, a staticmethod, or a property, the
    ``@deprecated`` decorator should go *under* ``@classmethod`` and
    ``@staticmethod`` (i.e., `deprecated` should directly decorate the
    underlying callable), but *over* ``@property``.

    When deprecating a class ``C`` intended to be used as a base class in a
    multiple inheritance hierarchy, ``C`` *must* define an ``__init__`` method
    (if ``C`` instead inherited its ``__init__`` from its own base class, then
    ``@deprecated`` would mess up ``__init__`` inheritance when installing its
    own (deprecation-emitting) ``C.__init__``).

    Parameters are the same as for `warn_deprecated`, except that *obj_type*
    defaults to 'class' if decorating a class, 'attribute' if decorating a
    property, and 'function' otherwise.

    Arguments:
        since : str
            The release at which this API became deprecated.
        message : str, optional
            Override the default deprecation message. The %(since)s,
            %(name)s, %(alternative)s, %(obj_type)s, %(addendum)s,
            and %(removal)s format specifiers will be replaced by the
            values of the respective arguments passed to this function.
        name : str, optional
            The name of the deprecated object.
        alternative : str, optional
            An alternative API that the user may use in place of the
            deprecated API. The deprecation warning will tell the user
            about this alternative if provided.
        pending : bool, optional
            If True, uses a PendingDeprecationWarning instead of a
            DeprecationWarning. Cannot be used together with removal.
        obj_type : str, optional
            The object type being deprecated.
        addendum : str, optional
            Additional text appended directly to the final message.
        removal : str, optional
            The expected removal version. With the default (an empty
            string), a removal version is automatically computed from
            since. Set to other Falsy values to not schedule a removal
            date. Cannot be used together with pending.

    Examples
    --------

        .. code-block:: python

            @deprecated('1.4.0')
            def the_function_to_deprecate():
                pass
    """

    def deprecate(
        obj: T,
        *,
        _obj_type: str = obj_type,
        _name: str = name,
        _message: str = message,
        _alternative: str = alternative,
        _alternative_import: str = alternative_import,
        _pending: bool = pending,
        _addendum: str = addendum,
    ) -> T:
        """Implementation of the decorator returned by `deprecated`."""

        def emit_warning() -> None:
            """Emit the warning."""
            warn_deprecated(
                since,
                message=_message,
                name=_name,
                alternative=_alternative,
                alternative_import=_alternative_import,
                pending=_pending,
                obj_type=_obj_type,
                addendum=_addendum,
                removal=removal,
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
            wrapped = obj.__init__  # type: ignore
            _name = _name or (
                f"{obj.__module__}.{obj.__name__}" if obj.__module__ else obj.__name__
            )
            old_doc = obj.__doc__

            def finalize(_: Any, new_doc: str) -> T:
                """Finalize the deprecation of a class."""
                try:
                    obj.__doc__ = new_doc
                except AttributeError:  # Can't set on some extension objects.
                    pass

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
            _name = _name or obj.fget.__name__
            old_doc = obj.__doc__

            class _deprecated_property(type(obj)):  # type: ignore
                """A deprecated property."""

                def __get__(self, instance, owner=None):  # type: ignore
                    if instance is not None or owner is not None:
                        emit_warning()
                    return super().__get__(instance, owner)

                def __set__(self, instance, value):  # type: ignore
                    if instance is not None:
                        emit_warning()
                    return super().__set__(instance, value)

                def __delete__(self, instance):  # type: ignore
                    if instance is not None:
                        emit_warning()
                    return super().__delete__(instance)

                def __set_name__(self, owner, set_name):  # type: ignore
                    nonlocal _name
                    if _name == "<lambda>":
                        _name = set_name

            def finalize(_: Any, new_doc: str) -> Any:  # type: ignore
                """Finalize the property."""
                return _deprecated_property(
                    fget=obj.fget, fset=obj.fset, fdel=obj.fdel, doc=new_doc
                )

        else:
            if not _obj_type:
                _obj_type = "function"
            wrapped = obj
            _name = _name or obj.__name__  # type: ignore
            old_doc = wrapped.__doc__

            def finalize(  # type: ignore
                wrapper: Callable[..., Any], new_doc: str
            ) -> T:
                """Wrap the wrapped function using the wrapper and update the docstring.

                Args:
                    wrapper: The wrapper function.
                    new_doc: The new docstring.

                Returns:
                    The wrapped function.
                """
                wrapper = functools.wraps(wrapped)(wrapper)
                wrapper.__doc__ = new_doc
                return wrapper

        old_doc = inspect.cleandoc(old_doc or "").strip("\n")

        # old_doc can be None
        if not old_doc:
            old_doc = ""

        # Modify the docstring to include a deprecation notice.
        notes_header = "\nNotes\n-----"
        components = [
            message,
            f"Use {alternative} instead." if alternative else "",
            addendum,
        ]
        details = " ".join([component.strip() for component in components if component])
        package = _name.split(".")[0].replace("_", "-") if "." in _name else None
        since_str = f"{package}=={since}" if package else since
        new_doc = (
            f"[*Deprecated*] {old_doc}\n"
            f"{notes_header if notes_header not in old_doc else ''}\n"
            f".. deprecated:: {since_str}\n"
            f"   {details}"
        )

        if inspect.iscoroutinefunction(obj):
            return finalize(awarning_emitting_wrapper, new_doc)
        else:
            return finalize(warning_emitting_wrapper, new_doc)

    return deprecate


@contextlib.contextmanager
def suppress_langchain_deprecation_warning() -> Generator[None, None, None]:
    """Context manager to suppress LangChainDeprecationWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LangChainDeprecationWarning)
        warnings.simplefilter("ignore", LangChainPendingDeprecationWarning)
        yield


def warn_deprecated(
    since: str,
    *,
    message: str = "",
    name: str = "",
    alternative: str = "",
    alternative_import: str = "",
    pending: bool = False,
    obj_type: str = "",
    addendum: str = "",
    removal: str = "",
) -> None:
    """Display a standardized deprecation.

    Arguments:
        since : str
            The release at which this API became deprecated.
        message : str, optional
            Override the default deprecation message. The %(since)s,
            %(name)s, %(alternative)s, %(obj_type)s, %(addendum)s,
            and %(removal)s format specifiers will be replaced by the
            values of the respective arguments passed to this function.
        name : str, optional
            The name of the deprecated object.
        alternative : str, optional
            An alternative API that the user may use in place of the
            deprecated API. The deprecation warning will tell the user
            about this alternative if provided.
        pending : bool, optional
            If True, uses a PendingDeprecationWarning instead of a
            DeprecationWarning. Cannot be used together with removal.
        obj_type : str, optional
            The object type being deprecated.
        addendum : str, optional
            Additional text appended directly to the final message.
        removal : str, optional
            The expected removal version. With the default (an empty
            string), a removal version is automatically computed from
            since. Set to other Falsy values to not schedule a removal
            date. Cannot be used together with pending.
    """
    if pending and removal:
        raise ValueError("A pending deprecation cannot have a scheduled removal")
    if alternative and alternative_import:
        raise ValueError("Cannot specify both alternative and alternative_import")
    if alternative_import and "." not in alternative_import:
        raise ValueError("alternative_import must be a fully qualified module path")

    if not pending:
        if not removal:
            removal = f"in {removal}" if removal else "within ?? minor releases"
            raise NotImplementedError(
                f"Need to determine which default deprecation schedule to use. "
                f"{removal}"
            )
        else:
            removal = f"in {removal}"

    if not message:
        message = ""
        package = name.split(".")[0].replace("_", "-") if "." in name else "LangChain"

        if obj_type:
            message += f"The {obj_type} `{name}`"
        else:
            message += f"`{name}`"

        if pending:
            message += " will be deprecated in a future version"
        else:
            message += f" was deprecated in {package} {since}"

            if removal:
                message += f" and will be removed {removal}"

        if alternative_import:
            alt_package = alternative_import.split(".")[0].replace("_", "-")
            if alt_package == package:
                message += f". Use {alternative_import} instead."
            else:
                alt_module, alt_name = alternative_import.rsplit(".", 1)
                message += (
                    f". An updated version of the {obj_type} exists in the "
                    f"{alt_package} package and should be used instead. To use it run "
                    f"`pip install -U {alt_package}` and import as "
                    f"`from {alt_module} import {alt_name}`."
                )
        elif alternative:
            message += f". Use {alternative} instead."

        if addendum:
            message += f" {addendum}"

    warning_cls = (
        LangChainPendingDeprecationWarning if pending else LangChainDeprecationWarning
    )
    warning = warning_cls(message)
    warnings.warn(warning, category=LangChainDeprecationWarning, stacklevel=2)


def surface_langchain_deprecation_warnings() -> None:
    """Unmute LangChain deprecation warnings."""
    warnings.filterwarnings(
        "default",
        category=LangChainPendingDeprecationWarning,
    )

    warnings.filterwarnings(
        "default",
        category=LangChainDeprecationWarning,
    )
