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
from typing import Any, Callable, Generator, Type, TypeVar


class LangChainBetaWarning(DeprecationWarning):
    """A class for issuing beta warnings for LangChain users."""


# PUBLIC API


T = TypeVar("T", Type, Callable)


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

    Arguments:
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

    Examples
    --------

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
        if isinstance(obj, type):
            if not _obj_type:
                _obj_type = "class"
            wrapped = obj.__init__  # type: ignore
            _name = _name or obj.__name__
            old_doc = obj.__doc__

            def finalize(wrapper: Callable[..., Any], new_doc: str) -> T:
                """Finalize the annotation of a class."""
                try:
                    obj.__doc__ = new_doc
                except AttributeError:  # Can't set on some extension objects.
                    pass
                obj.__init__ = functools.wraps(obj.__init__)(  # type: ignore[misc]
                    wrapper
                )
                return obj

        elif isinstance(obj, property):
            if not _obj_type:
                _obj_type = "attribute"
            wrapped = None
            _name = _name or obj.fget.__name__
            old_doc = obj.__doc__

            class _beta_property(type(obj)):  # type: ignore
                """A beta property."""

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
                return _beta_property(
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

        def emit_warning() -> None:
            """Emit the warning."""
            warn_beta(
                message=_message,
                name=_name,
                obj_type=_obj_type,
                addendum=_addendum,
            )

        def warning_emitting_wrapper(*args: Any, **kwargs: Any) -> Any:
            """Wrapper for the original wrapped callable that emits a warning.

            Args:
                *args: The positional arguments to the function.
                **kwargs: The keyword arguments to the function.

            Returns:
                The return value of the function being wrapped.
            """
            emit_warning()
            return wrapped(*args, **kwargs)

        old_doc = inspect.cleandoc(old_doc or "").strip("\n")

        if not old_doc:
            new_doc = "[*Beta*]"
        else:
            new_doc = f"[*Beta*]  {old_doc}"

        # Modify the docstring to include a beta notice.
        notes_header = "\nNotes\n-----"
        components = [
            message,
            addendum,
        ]
        details = " ".join([component.strip() for component in components if component])
        new_doc += (
            f"[*Beta*] {old_doc}\n"
            f"{notes_header if notes_header not in old_doc else ''}\n"
            f".. beta::\n"
            f"   {details}"
        )

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
    warnings.warn(warning, category=LangChainBetaWarning, stacklevel=2)


def surface_langchain_beta_warnings() -> None:
    """Unmute LangChain beta warnings."""
    warnings.filterwarnings(
        "default",
        category=LangChainBetaWarning,
    )
