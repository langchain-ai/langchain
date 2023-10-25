from langchain_cli.utils.git import _parse_dependency_string, DependencySource
from langchain_cli.constants import DEFAULT_GIT_REPO, DEFAULT_GIT_SUBDIRECTORY


def test_dependency_string() -> None:
    assert _parse_dependency_string(
        "git+ssh://git@github.com/efriis/myrepo.git"
    ) == DependencySource(
        git="ssh://git@github.com/efriis/myrepo.git",
        ref=None,
        subdirectory=None,
    )

    assert _parse_dependency_string(
        "git+https://github.com/efriis/myrepo.git#subdirectory=src"
    ) == DependencySource(
        git="https://github.com/efriis/myrepo.git",
        subdirectory="src",
        ref=None,
    )

    assert _parse_dependency_string(
        "git+ssh://git@github.com:efriis/myrepo.git#develop"
    ) == DependencySource(
        git="ssh://git@github.com:efriis/myrepo.git", ref="develop", subdirectory=None
    )

    assert _parse_dependency_string("simple-pirate") == DependencySource(
        git=DEFAULT_GIT_REPO,
        subdirectory=f"{DEFAULT_GIT_SUBDIRECTORY}/simple-pirate",
        ref=None,
    )
