"""Generic utility functions."""

import contextlib
import datetime
import functools
import importlib
import os
import warnings
from importlib.metadata import version
from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple, Union, overload

from packaging.version import parse
from requests import HTTPError, Response

from langchain_core.pydantic_v1 import SecretStr
from langchain_core.utils.pydantic import (
    is_pydantic_v1_subclass,
)


def xor_args(*arg_groups: Tuple[str, ...]) -> Callable:
    """Validate specified keyword args are mutually exclusive."

    Args:
        *arg_groups (Tuple[str, ...]): Groups of mutually exclusive keyword args.

    Returns:
        Callable: Decorator that validates the specified keyword args
            are mutually exclusive

    Raises:
        ValueError: If more than one arg in a group is defined.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Validate exactly one arg in each group is not None."""
            counts = [
                sum(1 for arg in arg_group if kwargs.get(arg) is not None)
                for arg_group in arg_groups
            ]
            invalid_groups = [i for i, count in enumerate(counts) if count != 1]
            if invalid_groups:
                invalid_group_names = [", ".join(arg_groups[i]) for i in invalid_groups]
                raise ValueError(
                    "Exactly one argument in each of the following"
                    " groups must be defined:"
                    f" {', '.join(invalid_group_names)}"
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def raise_for_status_with_text(response: Response) -> None:
    """Raise an error with the response text.

    Args:
        response (Response): The response to check for errors.

    Raises:
        ValueError: If the response has an error status code.
    """
    try:
        response.raise_for_status()
    except HTTPError as e:
        raise ValueError(response.text) from e


@contextlib.contextmanager
def mock_now(dt_value):  # type: ignore
    """Context manager for mocking out datetime.now() in unit tests.

    Args:
        dt_value: The datetime value to use for datetime.now().

    Yields:
        datetime.datetime: The mocked datetime class.

    Example:
    with mock_now(datetime.datetime(2011, 2, 3, 10, 11)):
        assert datetime.datetime.now() == datetime.datetime(2011, 2, 3, 10, 11)
    """

    class MockDateTime(datetime.datetime):
        """Mock datetime.datetime.now() with a fixed datetime."""

        @classmethod
        def now(cls):  # type: ignore
            # Create a copy of dt_value.
            return datetime.datetime(
                dt_value.year,
                dt_value.month,
                dt_value.day,
                dt_value.hour,
                dt_value.minute,
                dt_value.second,
                dt_value.microsecond,
                dt_value.tzinfo,
            )

    real_datetime = datetime.datetime
    datetime.datetime = MockDateTime
    try:
        yield datetime.datetime
    finally:
        datetime.datetime = real_datetime


def guard_import(
    module_name: str, *, pip_name: Optional[str] = None, package: Optional[str] = None
) -> Any:
    """Dynamically import a module and raise an exception if the module is not
    installed.

    Args:
        module_name (str): The name of the module to import.
        pip_name (str, optional): The name of the module to install with pip.
            Defaults to None.
        package (str, optional): The package to import the module from.
            Defaults to None.

    Returns:
        Any: The imported module.

    Raises:
        ImportError: If the module is not installed.
    """
    try:
        module = importlib.import_module(module_name, package)
    except (ImportError, ModuleNotFoundError) as e:
        pip_name = pip_name or module_name.split(".")[0].replace("_", "-")
        raise ImportError(
            f"Could not import {module_name} python package. "
            f"Please install it with `pip install {pip_name}`."
        ) from e
    return module


def check_package_version(
    package: str,
    lt_version: Optional[str] = None,
    lte_version: Optional[str] = None,
    gt_version: Optional[str] = None,
    gte_version: Optional[str] = None,
) -> None:
    """Check the version of a package.

    Args:
        package (str): The name of the package.
        lt_version (str, optional): The version must be less than this.
            Defaults to None.
        lte_version (str, optional): The version must be less than or equal to this.
            Defaults to None.
        gt_version (str, optional): The version must be greater than this.
            Defaults to None.
        gte_version (str, optional): The version must be greater than or equal to this.
            Defaults to None.

    Raises:
        ValueError: If the package version does not meet the requirements.
    """
    imported_version = parse(version(package))
    if lt_version is not None and imported_version >= parse(lt_version):
        raise ValueError(
            f"Expected {package} version to be < {lt_version}. Received "
            f"{imported_version}."
        )
    if lte_version is not None and imported_version > parse(lte_version):
        raise ValueError(
            f"Expected {package} version to be <= {lte_version}. Received "
            f"{imported_version}."
        )
    if gt_version is not None and imported_version <= parse(gt_version):
        raise ValueError(
            f"Expected {package} version to be > {gt_version}. Received "
            f"{imported_version}."
        )
    if gte_version is not None and imported_version < parse(gte_version):
        raise ValueError(
            f"Expected {package} version to be >= {gte_version}. Received "
            f"{imported_version}."
        )


def get_pydantic_field_names(pydantic_cls: Any) -> Set[str]:
    """Get field names, including aliases, for a pydantic class.

    Args:
        pydantic_cls: Pydantic class.

    Returns:
        Set[str]: Field names.
    """
    all_required_field_names = set()
    if is_pydantic_v1_subclass(pydantic_cls):
        for field in pydantic_cls.__fields__.values():
            all_required_field_names.add(field.name)
            if field.has_alias:
                all_required_field_names.add(field.alias)
    else:  # Assuming pydantic 2 for now
        for name, field in pydantic_cls.model_fields.items():
            all_required_field_names.add(name)
            if field.alias:
                all_required_field_names.add(field.alias)
    return all_required_field_names


def build_extra_kwargs(
    extra_kwargs: Dict[str, Any],
    values: Dict[str, Any],
    all_required_field_names: Set[str],
) -> Dict[str, Any]:
    """Build extra kwargs from values and extra_kwargs.

    Args:
        extra_kwargs: Extra kwargs passed in by user.
        values: Values passed in by user.
        all_required_field_names: All required field names for the pydantic class.

    Returns:
        Dict[str, Any]: Extra kwargs.

    Raises:
        ValueError: If a field is specified in both values and extra_kwargs.
        ValueError: If a field is specified in model_kwargs.
    """
    for field_name in list(values):
        if field_name in extra_kwargs:
            raise ValueError(f"Found {field_name} supplied twice.")
        if field_name not in all_required_field_names:
            warnings.warn(
                f"""WARNING! {field_name} is not default parameter.
                {field_name} was transferred to model_kwargs.
                Please confirm that {field_name} is what you intended.""",
                stacklevel=7,
            )
            extra_kwargs[field_name] = values.pop(field_name)

    invalid_model_kwargs = all_required_field_names.intersection(extra_kwargs.keys())
    if invalid_model_kwargs:
        raise ValueError(
            f"Parameters {invalid_model_kwargs} should be specified explicitly. "
            f"Instead they were passed in as part of `model_kwargs` parameter."
        )

    return extra_kwargs


def convert_to_secret_str(value: Union[SecretStr, str]) -> SecretStr:
    """Convert a string to a SecretStr if needed.

    Args:
        value (Union[SecretStr, str]): The value to convert.

    Returns:
        SecretStr: The SecretStr value.
    """
    if isinstance(value, SecretStr):
        return value
    return SecretStr(value)


class _NoDefaultType:
    """Type to indicate no default value is provided."""

    pass


_NoDefault = _NoDefaultType()


@overload
def from_env(key: str, /) -> Callable[[], str]: ...


@overload
def from_env(key: str, /, *, default: str) -> Callable[[], str]: ...


@overload
def from_env(key: Sequence[str], /, *, default: str) -> Callable[[], str]: ...


@overload
def from_env(key: str, /, *, error_message: str) -> Callable[[], str]: ...


@overload
def from_env(
    key: Union[str, Sequence[str]], /, *, default: str, error_message: Optional[str]
) -> Callable[[], str]: ...


@overload
def from_env(
    key: str, /, *, default: None, error_message: Optional[str]
) -> Callable[[], Optional[str]]: ...


@overload
def from_env(
    key: Union[str, Sequence[str]], /, *, default: None
) -> Callable[[], Optional[str]]: ...


def from_env(
    key: Union[str, Sequence[str]],
    /,
    *,
    default: Union[str, _NoDefaultType, None] = _NoDefault,
    error_message: Optional[str] = None,
) -> Union[Callable[[], str], Callable[[], Optional[str]]]:
    """Create a factory method that gets a value from an environment variable.

    Args:
        key: The environment variable to look up. If a list of keys is provided,
            the first key found in the environment will be used.
            If no key is found, the default value will be used if set,
            otherwise an error will be raised.
        default: The default value to return if the environment variable is not set.
        error_message: the error message which will be raised if the key is not found
            and no default value is provided.
            This will be raised as a ValueError.
    """

    def get_from_env_fn() -> Optional[str]:
        """Get a value from an environment variable."""
        if isinstance(key, (list, tuple)):
            for k in key:
                if k in os.environ:
                    return os.environ[k]
        if isinstance(key, str):
            if key in os.environ:
                return os.environ[key]

        if isinstance(default, (str, type(None))):
            return default
        else:
            if error_message:
                raise ValueError(error_message)
            else:
                raise ValueError(
                    f"Did not find {key}, please add an environment variable"
                    f" `{key}` which contains it, or pass"
                    f" `{key}` as a named parameter."
                )

    return get_from_env_fn


@overload
def secret_from_env(key: str, /) -> Callable[[], SecretStr]: ...


@overload
def secret_from_env(key: str, /, *, default: str) -> Callable[[], SecretStr]: ...


@overload
def secret_from_env(
    key: Union[str, Sequence[str]], /, *, default: None
) -> Callable[[], Optional[SecretStr]]: ...


@overload
def secret_from_env(key: str, /, *, error_message: str) -> Callable[[], SecretStr]: ...


def secret_from_env(
    key: Union[str, Sequence[str]],
    /,
    *,
    default: Union[str, _NoDefaultType, None] = _NoDefault,
    error_message: Optional[str] = None,
) -> Union[Callable[[], Optional[SecretStr]], Callable[[], SecretStr]]:
    """Secret from env.

    Args:
        key: The environment variable to look up.
        default: The default value to return if the environment variable is not set.
        error_message: the error message which will be raised if the key is not found
            and no default value is provided.
            This will be raised as a ValueError.

    Returns:
        factory method that will look up the secret from the environment.
    """

    def get_secret_from_env() -> Optional[SecretStr]:
        """Get a value from an environment variable."""
        if isinstance(key, (list, tuple)):
            for k in key:
                if k in os.environ:
                    return SecretStr(os.environ[k])
        if isinstance(key, str):
            if key in os.environ:
                return SecretStr(os.environ[key])
        if isinstance(default, str):
            return SecretStr(default)
        elif isinstance(default, type(None)):
            return None
        else:
            if error_message:
                raise ValueError(error_message)
            else:
                raise ValueError(
                    f"Did not find {key}, please add an environment variable"
                    f" `{key}` which contains it, or pass"
                    f" `{key}` as a named parameter."
                )

    return get_secret_from_env
