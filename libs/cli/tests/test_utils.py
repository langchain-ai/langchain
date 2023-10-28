from typing import Dict, Optional

import pytest

from langchain_cli.constants import (
    DEFAULT_GIT_REF,
    DEFAULT_GIT_REPO,
    DEFAULT_GIT_SUBDIRECTORY,
)
from langchain_cli.utils.git import DependencySource, parse_dependency_string


def _assert_dependency_equals(
    dep: DependencySource,
    *,
    git: Optional[str] = None,
    ref: Optional[str] = None,
    subdirectory: Optional[str] = None,
    event_metadata: Optional[Dict] = None,
) -> None:
    assert dep["git"] == git
    assert dep["ref"] == ref
    assert dep["subdirectory"] == subdirectory
    if event_metadata is not None:
        assert dep["event_metadata"] == event_metadata


def test_dependency_string() -> None:
    _assert_dependency_equals(
        parse_dependency_string(
            "git+ssh://git@github.com/efriis/myrepo.git", None, None, None
        ),
        git="ssh://git@github.com/efriis/myrepo.git",
        ref=None,
        subdirectory=None,
    )

    _assert_dependency_equals(
        parse_dependency_string(
            "git+https://github.com/efriis/myrepo.git#subdirectory=src",
            None,
            None,
            None,
        ),
        git="https://github.com/efriis/myrepo.git",
        subdirectory="src",
        ref=None,
    )

    _assert_dependency_equals(
        parse_dependency_string(
            "git+ssh://git@github.com:efriis/myrepo.git#develop", None, None, None
        ),
        git="ssh://git@github.com:efriis/myrepo.git",
        ref="develop",
        subdirectory=None,
    )

    # also support a slash in ssh
    _assert_dependency_equals(
        parse_dependency_string(
            "git+ssh://git@github.com/efriis/myrepo.git#develop", None, None, None
        ),
        git="ssh://git@github.com/efriis/myrepo.git",
        ref="develop",
        subdirectory=None,
    )

    # looks like poetry supports both an @ and a #
    _assert_dependency_equals(
        parse_dependency_string(
            "git+ssh://git@github.com:efriis/myrepo.git@develop", None, None, None
        ),
        git="ssh://git@github.com:efriis/myrepo.git",
        ref="develop",
        subdirectory=None,
    )

    _assert_dependency_equals(
        parse_dependency_string("simple-pirate", None, None, None),
        git=DEFAULT_GIT_REPO,
        subdirectory=f"{DEFAULT_GIT_SUBDIRECTORY}/simple-pirate",
        ref=DEFAULT_GIT_REF,
    )


def test_dependency_string_both() -> None:
    _assert_dependency_equals(
        parse_dependency_string(
            "git+https://github.com/efriis/myrepo.git@branch#subdirectory=src",
            None,
            None,
            None,
        ),
        git="https://github.com/efriis/myrepo.git",
        subdirectory="src",
        ref="branch",
    )


def test_dependency_string_invalids() -> None:
    # expect error for wrong order
    with pytest.raises(ValueError):
        parse_dependency_string(
            "git+https://github.com/efriis/myrepo.git#subdirectory=src@branch",
            None,
            None,
            None,
        )
    # expect error for @subdirectory


def test_dependency_string_edge_case() -> None:
    # weird unsolvable edge case of
    # git+ssh://a@b
    # this could be a ssh dep with user=a, and default ref
    # or a ssh dep at a with ref=b.
    # in this case, assume the first case (be greedy with the '@')
    _assert_dependency_equals(
        parse_dependency_string("git+ssh://a@b", None, None, None),
        git="ssh://a@b",
        subdirectory=None,
        ref=None,
    )

    # weird one that is actually valid
    _assert_dependency_equals(
        parse_dependency_string(
            "git+https://github.com/efriis/myrepo.git@subdirectory=src",
            None,
            None,
            None,
        ),
        git="https://github.com/efriis/myrepo.git",
        subdirectory=None,
        ref="subdirectory=src",
    )
