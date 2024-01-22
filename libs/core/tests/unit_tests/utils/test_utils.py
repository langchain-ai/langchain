from typing import Dict, Optional, Tuple, Type
from unittest.mock import patch

import pytest

from langchain_core.utils import check_package_version


@pytest.mark.parametrize(
    ("package", "check_kwargs", "actual_version", "expected"),
    [
        ("stub", {"gt_version": "0.1"}, "0.1.2", None),
        ("stub", {"gt_version": "0.1.2"}, "0.1.12", None),
        ("stub", {"gt_version": "0.1.2"}, "0.1.2", (ValueError, "> 0.1.2")),
        ("stub", {"gte_version": "0.1"}, "0.1.2", None),
        ("stub", {"gte_version": "0.1.2"}, "0.1.2", None),
    ],
)
def test_check_package_version(
    package: str,
    check_kwargs: Dict[str, Optional[str]],
    actual_version: str,
    expected: Optional[Tuple[Type[Exception], str]],
) -> None:
    with patch("langchain_core.utils.utils.version", return_value=actual_version):
        if expected is None:
            check_package_version(package, **check_kwargs)
        else:
            with pytest.raises(expected[0], match=expected[1]):
                check_package_version(package, **check_kwargs)
