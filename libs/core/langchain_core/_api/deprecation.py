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
from collections.abc import Generator
from typing import (
    Any,
    Callable,
    TypeVar,
    Union,
    cast,
)

from pydantic.fields import FieldInfo
from pydantic.v1.fields import FieldInfo as FieldInfoV1
from typing_extensions import ParamSpec

from langchain_core._api.internal import is_caller_internal


class LangChainDeprecationWarning(DeprecationWarning):
    """A class for issuing deprecation warnings for LangChain users."""


class LangChainPendingDeprecationWarning(PendingDeprecationWarning):
    """A class for issuing deprecation warnings for LangChain users."""


# PUBLIC API


# Last Any should be FieldInfoV1 but this leads to circular imports
T = TypeVar("T", bound=Union[type, Callable[..., Any], Any])


def _validate_deprecation_params(
    removal: str,
    alternative: str,
    alternative_import: str,
    *,
    pending: bool,
) -> None:
    """Validate the deprecation parameters."""
    if pending and removal:
        msg = "A pending deprecation cannot have a scheduled removal"
        raise ValueError(msg)
    if alternative and alternative_import:
        msg = "Cannot specify both alternative and alternative_import"
        raise ValueError(msg)

    if alternative_import and "." not in alternative_import:
        msg = (
            "alternative_import must be a fully qualified module path. Got "
            f" {alternative_import}"
        )
        raise ValueError(msg)


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
    package: str = "",
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

    Args:
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
        alternative_import: str, optional
            An alternative import that the user may use instead.
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
        package: str, optional
            The package of the deprecated object.

    Returns:
        A decorator to mark a function or class as deprecated.

    Examples:

        .. code-block:: python

            @deprecated("1.4.0")
            def the_function_to_deprecate():
                pass

    """
    _validate_deprecation_params(
        removal, alternative, alternative_import, pending=pending
    )

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
        _package: str = package,
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
                package=_package,
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

        _package = _package or obj.__module__.split(".")[0].replace("_", "-")

        if isinstance(obj, type):
            if not _obj_type:
                _obj_type = "class"
            wrapped = obj.__init__  # type: ignore[misc]
            _name = _name or obj.__qualname__
            old_doc = obj.__doc__

            def finalize(wrapper: Callable[..., Any], new_doc: str) -> T:  # noqa: ARG001
                """Finalize the deprecation of a class."""
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

        elif isinstance(obj, FieldInfoV1):
            wrapped = None
            if not _obj_type:
                _obj_type = "attribute"
            if not _name:
                msg = f"Field {obj} must have a name to be deprecated."
                raise ValueError(msg)
            old_doc = obj.description

            def finalize(wrapper: Callable[..., Any], new_doc: str) -> T:  # noqa: ARG001
                return cast(
                    "T",
                    FieldInfoV1(
                        default=obj.default,
                        default_factory=obj.default_factory,
                        description=new_doc,
                        alias=obj.alias,
                        exclude=obj.exclude,
                    ),
                )

        elif isinstance(obj, FieldInfo):
            wrapped = None
            if not _obj_type:
                _obj_type = "attribute"
            if not _name:
                msg = f"Field {obj} must have a name to be deprecated."
                raise ValueError(msg)
            old_doc = obj.description

            def finalize(wrapper: Callable[..., Any], new_doc: str) -> T:  # noqa: ARG001
                return cast(
                    "T",
                    FieldInfo(
                        default=obj.default,
                        default_factory=obj.default_factory,
                        description=new_doc,
                        alias=obj.alias,
                        exclude=obj.exclude,
                    ),
                )

        elif isinstance(obj, property):
            if not _obj_type:
                _obj_type = "attribute"
            wrapped = None
            _name = _name or cast("Union[type, Callable]", obj.fget).__qualname__
            old_doc = obj.__doc__

            class _DeprecatedProperty(property):
                """A deprecated property."""

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

                def __get__(
                    self, instance: Any, owner: Union[type, None] = None
                ) -> Any:
                    if instance is not None or owner is not None:
                        emit_warning()
                    if self.fget is None:
                        return None
                    return self.fget(instance)

                def __set__(self, instance: Any, value: Any) -> None:
                    if instance is not None:
                        emit_warning()
                    if self.fset is not None:
                        self.fset(instance, value)

                def __delete__(self, instance: Any) -> None:
                    if instance is not None:
                        emit_warning()
                    if self.fdel is not None:
                        self.fdel(instance)

                def __set_name__(self, owner: Union[type, None], set_name: str) -> None:
                    nonlocal _name
                    if _name == "<lambda>":
                        _name = set_name

            def finalize(wrapper: Callable[..., Any], new_doc: str) -> T:  # noqa: ARG001
                """Finalize the property."""
                return cast(
                    "T",
                    _DeprecatedProperty(
                        fget=obj.fget, fset=obj.fset, fdel=obj.fdel, doc=new_doc
                    ),
                )

        else:
            _name = _name or cast("Union[type, Callable]", obj).__qualname__
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

        old_doc = inspect.cleandoc(old_doc or "").strip("\n")

        # old_doc can be None
        if not old_doc:
            old_doc = ""

        # Modify the docstring to include a deprecation notice.
        if (
            _alternative
            and _alternative.rsplit(".", maxsplit=1)[-1].lower()
            == _alternative.rsplit(".", maxsplit=1)[-1]
        ):
            _alternative = f":meth:`~{_alternative}`"
        elif _alternative:
            _alternative = f":class:`~{_alternative}`"

        if (
            _alternative_import
            and _alternative_import.rsplit(".", maxsplit=1)[-1].lower()
            == _alternative_import.rsplit(".", maxsplit=1)[-1]
        ):
            _alternative_import = f":meth:`~{_alternative_import}`"
        elif _alternative_import:
            _alternative_import = f":class:`~{_alternative_import}`"

        components = [
            _message,
            f"Use {_alternative} instead." if _alternative else "",
            f"Use ``{_alternative_import}`` instead." if _alternative_import else "",
            _addendum,
        ]
        details = " ".join([component.strip() for component in components if component])
        package = _package or (
            _name.split(".")[0].replace("_", "-") if "." in _name else None
        )
        if removal:
            if removal.startswith("1.") and package and package.startswith("langchain"):
                removal_str = f"It will not be removed until {package}=={removal}."
            else:
                removal_str = f"It will be removed in {package}=={removal}."
        else:
            removal_str = ""
        new_doc = f"""\
.. deprecated:: {since} {details} {removal_str}

{old_doc}\
"""

        if inspect.iscoroutinefunction(obj):
            return finalize(awarning_emitting_wrapper, new_doc)
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
    package: str = "",
) -> None:
    """Display a standardized deprecation.

    Args:
        since:
            The release at which this API became deprecated.
        message:
            Override the default deprecation message. The %(since)s,
            %(name)s, %(alternative)s, %(obj_type)s, %(addendum)s,
            and %(removal)s format specifiers will be replaced by the
            values of the respective arguments passed to this function.
        name:
            The name of the deprecated object.
        alternative:
            An alternative API that the user may use in place of the
            deprecated API. The deprecation warning will tell the user
            about this alternative if provided.
        alternative_import:
            An alternative import that the user may use instead.
        pending:
            If True, uses a PendingDeprecationWarning instead of a
            DeprecationWarning. Cannot be used together with removal.
        obj_type:
            The object type being deprecated.
        addendum:
            Additional text appended directly to the final message.
        removal:
            The expected removal version. With the default (an empty
            string), a removal version is automatically computed from
            since. Set to other Falsy values to not schedule a removal
            date. Cannot be used together with pending.
        package:
            The package of the deprecated object.
    """
    if not pending:
        if not removal:
            removal = f"in {removal}" if removal else "within ?? minor releases"
            msg = (
                f"Need to determine which default deprecation schedule to use. "
                f"{removal}"
            )
            raise NotImplementedError(msg)
        removal = f"in {removal}"

    if not message:
        message = ""
        package_ = (
            package or name.split(".", maxsplit=1)[0].replace("_", "-")
            if "." in name
            else "LangChain"
        )

        if obj_type:
            message += f"The {obj_type} `{name}`"
        else:
            message += f"`{name}`"

        if pending:
            message += " will be deprecated in a future version"
        else:
            message += f" was deprecated in {package_} {since}"

            if removal:
                message += f" and will be removed {removal}"

        if alternative_import:
            alt_package = alternative_import.split(".", maxsplit=1)[0].replace("_", "-")
            if alt_package == package_:
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
    warnings.warn(warning, category=LangChainDeprecationWarning, stacklevel=4)


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


_P = ParamSpec("_P")
_R = TypeVar("_R")


def rename_parameter(
    *,
    since: str,
    removal: str,
    old: str,
    new: str,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Decorator indicating that parameter *old* of *func* is renamed to *new*.

    The actual implementation of *func* should use *new*, not *old*.  If *old*
    is passed to *func*, a DeprecationWarning is emitted, and its value is
    used, even if *new* is also passed by keyword.

    Args:
        since: The version in which the parameter was renamed.
        removal: The version in which the old parameter will be removed.
        old: The old parameter name.
        new: The new parameter name.

    Returns:
        A decorator indicating that a parameter was renamed.

    Example:

        .. code-block:: python

            @_api.rename_parameter("3.1", "bad_name", "good_name")
            def func(good_name): ...

    """

    def decorator(f: Callable[_P, _R]) -> Callable[_P, _R]:
        @functools.wraps(f)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            if new in kwargs and old in kwargs:
                msg = f"{f.__name__}() got multiple values for argument {new!r}"
                raise TypeError(msg)
            if old in kwargs:
                warn_deprecated(
                    since,
                    removal=removal,
                    message=f"The parameter `{old}` of `{f.__name__}` was "
                    f"deprecated in {since} and will be removed "
                    f"in {removal} Use `{new}` instead.",
                )
                kwargs[new] = kwargs.pop(old)
            return f(*args, **kwargs)

        return wrapper

    return decorator
