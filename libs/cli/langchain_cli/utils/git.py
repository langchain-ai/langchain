import hashlib
import re
import shutil
from pathlib import Path
from typing import Optional, TypedDict

from git import Repo

from langchain_cli.constants import (
    DEFAULT_GIT_REF,
    DEFAULT_GIT_REPO,
    DEFAULT_GIT_SUBDIRECTORY,
)


class DependencySource(TypedDict):
    git: str
    ref: Optional[str]
    subdirectory: Optional[str]


# use poetry dependency string format
def parse_dependency_string(package_string: str) -> DependencySource:
    if package_string.startswith("git+"):
        # remove git+
        gitstring = package_string[4:]
        subdirectory = None
        ref = None
        # first check for #subdirectory= on the end
        if "#subdirectory=" in gitstring:
            gitstring, subdirectory = gitstring.split("#subdirectory=")
            if "#" in subdirectory or "@" in subdirectory:
                raise ValueError(
                    "#subdirectory must be the last part of the dependency string"
                )

        # find first slash after ://
        # find @ or # after that slash
        # remainder is ref
        # if no @ or #, then ref is None

        # find first slash after ://
        if "://" not in gitstring:
            raise ValueError(
                "git+ dependencies must start with git+https:// or git+ssh://"
            )

        _, find_slash = gitstring.split("://", 1)

        if "/" not in find_slash:
            post_slash = find_slash
            ref = None
        else:
            _, post_slash = find_slash.split("/", 1)
            if "@" in post_slash or "#" in post_slash:
                _, ref = re.split(r"[@#]", post_slash, 1)

        # gitstring is everything before that
        gitstring = gitstring[: -len(ref) - 1] if ref is not None else gitstring

        return DependencySource(
            git=gitstring,
            ref=ref,
            subdirectory=subdirectory,
        )

    elif package_string.startswith("https://"):
        raise NotImplementedError("url dependencies are not supported yet")
    else:
        # it's a default git repo dependency
        subdirectory = str(Path(DEFAULT_GIT_SUBDIRECTORY) / package_string)
        return DependencySource(
            git=DEFAULT_GIT_REPO, ref=DEFAULT_GIT_REF, subdirectory=subdirectory
        )


def _get_repo_path(gitstring: str, repo_dir: Path) -> Path:
    # only based on git for now
    hashed = hashlib.sha256(gitstring.encode("utf-8")).hexdigest()[:8]

    removed_protocol = gitstring.split("://")[-1]
    removed_basename = re.split(r"[/:]", removed_protocol, 1)[-1]
    removed_extras = removed_basename.split("#")[0]
    foldername = re.sub(r"[^a-zA-Z0-9_]", "_", removed_extras)

    directory_name = f"{foldername}_{hashed}"
    return repo_dir / directory_name


def update_repo(gitstring: str, ref: Optional[str], repo_dir: Path) -> Path:
    # see if path already saved
    repo_path = _get_repo_path(gitstring, repo_dir)
    if repo_path.exists():
        shutil.rmtree(repo_path)

    # now we have fresh dir
    Repo.clone_from(gitstring, repo_path, branch=ref, depth=1)
    return repo_path


def copy_repo(
    source: Path,
    destination: Path,
) -> None:
    """
    Copies a repo, ignoring git folders.

    Raises FileNotFound error if it can't find source
    """

    def ignore_func(_, files):
        return [f for f in files if f == ".git"]

    shutil.copytree(source, destination, ignore=ignore_func)
