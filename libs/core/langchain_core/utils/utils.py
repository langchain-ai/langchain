"""Generic utility functions."""
import contextlib
import datetime
import functools
import importlib
import warnings
from importlib.metadata import version
from typing import Any, Callable, Dict, Optional, Set, Tuple, Union

from packaging.version import parse
from requests import HTTPError, Response

from langchain_core.pydantic_v1 import SecretStr


def xor_args(*arg_groups: Tuple[str, ...]) -> Callable:
    """Validate specified keyword args are mutually exclusive."""

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
    """Raise an error with the response text."""
    try:
        response.raise_for_status()
    except HTTPError as e:
        raise ValueError(response.text) from e


@contextlib.contextmanager
def mock_now(dt_value):  # type: ignore
    """Context manager for mocking out datetime.now() in unit tests.

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
    """Dynamically imports a module and raises a helpful exception if the module is not
    installed."""
    try:
        module = importlib.import_module(module_name, package)
    except ImportError:
        raise ImportError(
            f"Could not import {module_name} python package. "
            f"Please install it with `pip install {pip_name or module_name}`."
        )
    return module


def check_package_version(
    package: str,
    lt_version: Optional[str] = None,
    lte_version: Optional[str] = None,
    gt_version: Optional[str] = None,
    gte_version: Optional[str] = None,
) -> None:
    """Check the version of a package."""
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
        pydantic_cls: Pydantic class."""
    all_required_field_names = set()
    for field in pydantic_cls.__fields__.values():
        all_required_field_names.add(field.name)
        if field.has_alias:
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
    """
    for field_name in list(values):
        if field_name in extra_kwargs:
            raise ValueError(f"Found {field_name} supplied twice.")
        if field_name not in all_required_field_names:
            warnings.warn(
                f"""WARNING! {field_name} is not default parameter.
                {field_name} was transferred to model_kwargs.
                Please confirm that {field_name} is what you intended."""
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
    """Convert a string to a SecretStr if needed."""
    if isinstance(value, SecretStr):
        return value
    return SecretStr(value)
