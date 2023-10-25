from langc.utils.git import _parse_dependency_string, DependencySource
from langc.constants import DEFAULT_GIT_REPO, DEFAULT_GIT_BRANCH


def test_dependency_string() -> None:
    assert _parse_dependency_string(
        "git+ssh://git@github.com/efriis/myrepo.git"
    ) == DependencySource(
        git="ssh://git@github.com/efriis/myrepo.git",
        ref=DEFAULT_GIT_BRANCH,
        subdirectory=None,
    )

    assert _parse_dependency_string(
        "git+https://github.com/efriis/myrepo.git#subdirectory=src"
    ) == DependencySource(
        git="https://github.com/efriis/myrepo.git",
        subdirectory="src",
        ref=DEFAULT_GIT_BRANCH,
    )

    assert _parse_dependency_string(
        "git+ssh://git@github.com:efriis/myrepo.git#develop"
    ) == DependencySource(
        git="ssh://git@github.com:efriis/myrepo.git", ref="develop", subdirectory=None
    )

    assert _parse_dependency_string("simple-pirate") == DependencySource(
        git=DEFAULT_GIT_REPO, subdirectory="simple-pirate", ref=DEFAULT_GIT_BRANCH
    )
