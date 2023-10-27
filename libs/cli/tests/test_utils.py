import pytest

from langchain_cli.constants import (
    DEFAULT_GIT_REF,
    DEFAULT_GIT_REPO,
    DEFAULT_GIT_SUBDIRECTORY,
)
from langchain_cli.utils.git import DependencySource, parse_dependency_string


def test_dependency_string() -> None:
    assert parse_dependency_string(
        "git+ssh://git@github.com/efriis/myrepo.git"
    ) == DependencySource(
        git="ssh://git@github.com/efriis/myrepo.git",
        ref=None,
        subdirectory=None,
    )

    assert parse_dependency_string(
        "git+https://github.com/efriis/myrepo.git#subdirectory=src"
    ) == DependencySource(
        git="https://github.com/efriis/myrepo.git",
        subdirectory="src",
        ref=None,
    )

    assert parse_dependency_string(
        "git+ssh://git@github.com:efriis/myrepo.git#develop"
    ) == DependencySource(
        git="ssh://git@github.com:efriis/myrepo.git", ref="develop", subdirectory=None
    )

    # also support a slash in ssh
    assert parse_dependency_string(
        "git+ssh://git@github.com/efriis/myrepo.git#develop"
    ) == DependencySource(
        git="ssh://git@github.com/efriis/myrepo.git", ref="develop", subdirectory=None
    )

    # looks like poetry supports both an @ and a #
    assert parse_dependency_string(
        "git+ssh://git@github.com:efriis/myrepo.git@develop"
    ) == DependencySource(
        git="ssh://git@github.com:efriis/myrepo.git", ref="develop", subdirectory=None
    )

    assert parse_dependency_string("simple-pirate") == DependencySource(
        git=DEFAULT_GIT_REPO,
        subdirectory=f"{DEFAULT_GIT_SUBDIRECTORY}/simple-pirate",
        ref=DEFAULT_GIT_REF,
    )


def test_dependency_string_both() -> None:
    assert parse_dependency_string(
        "git+https://github.com/efriis/myrepo.git@branch#subdirectory=src"
    ) == DependencySource(
        git="https://github.com/efriis/myrepo.git",
        subdirectory="src",
        ref="branch",
    )


def test_dependency_string_invalids() -> None:
    # expect error for wrong order
    with pytest.raises(ValueError):
        parse_dependency_string(
            "git+https://github.com/efriis/myrepo.git#subdirectory=src@branch"
        )
    # expect error for @subdirectory


def test_dependency_string_edge_case() -> None:
    # weird unsolvable edge case of
    # git+ssh://a@b
    # this could be a ssh dep with user=a, and default ref
    # or a ssh dep at a with ref=b.
    # in this case, assume the first case (be greedy with the '@')
    assert parse_dependency_string("git+ssh://a@b") == DependencySource(
        git="ssh://a@b",
        subdirectory=None,
        ref=None,
    )

    # weird one that is actually valid
    assert parse_dependency_string(
        "git+https://github.com/efriis/myrepo.git@subdirectory=src"
    ) == DependencySource(
        git="https://github.com/efriis/myrepo.git",
        subdirectory=None,
        ref="subdirectory=src",
    )
