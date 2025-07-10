import hashlib
import re
import shutil
from collections.abc import Sequence
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
    api_path: Optional[str]
    event_metadata: dict


# use poetry dependency string format
def parse_dependency_string(
    dep: Optional[str],
    repo: Optional[str],
    branch: Optional[str],
    api_path: Optional[str],
) -> DependencySource:
    if dep is not None and dep.startswith("git+"):
        if repo is not None or branch is not None:
            msg = (
                "If a dependency starts with git+, you cannot manually specify "
                "a repo or branch."
            )
            raise ValueError(msg)
        # remove git+
        gitstring = dep[4:]
        subdirectory = None
        ref = None
        # first check for #subdirectory= on the end
        if "#subdirectory=" in gitstring:
            gitstring, subdirectory = gitstring.split("#subdirectory=")
            if "#" in subdirectory or "@" in subdirectory:
                msg = "#subdirectory must be the last part of the dependency string"
                raise ValueError(msg)

        # find first slash after ://
        # find @ or # after that slash
        # remainder is ref
        # if no @ or #, then ref is None

        # find first slash after ://
        if "://" not in gitstring:
            msg = "git+ dependencies must start with git+https:// or git+ssh://"
            raise ValueError(msg)

        _, find_slash = gitstring.split("://", 1)

        if "/" not in find_slash:
            post_slash = find_slash
            ref = None
        else:
            _, post_slash = find_slash.split("/", 1)
            if "@" in post_slash or "#" in post_slash:
                _, ref = re.split(r"[@#]", post_slash, maxsplit=1)

        # gitstring is everything before that
        gitstring = gitstring[: -len(ref) - 1] if ref is not None else gitstring

        return DependencySource(
            git=gitstring,
            ref=ref,
            subdirectory=subdirectory,
            api_path=api_path,
            event_metadata={"dependency_string": dep},
        )

    if dep is not None and dep.startswith("https://"):
        msg = "Only git dependencies are supported"
        raise ValueError(msg)
    # if repo is none, use default, including subdirectory
    base_subdir = Path(DEFAULT_GIT_SUBDIRECTORY) if repo is None else Path()
    subdir = str(base_subdir / dep) if dep is not None else None
    gitstring = (
        DEFAULT_GIT_REPO
        if repo is None
        else f"https://github.com/{repo.strip('/')}.git"
    )
    ref = DEFAULT_GIT_REF if branch is None else branch
    # it's a default git repo dependency
    return DependencySource(
        git=gitstring,
        ref=ref,
        subdirectory=subdir,
        api_path=api_path,
        event_metadata={
            "dependency_string": dep,
            "used_repo_flag": repo is not None,
            "used_branch_flag": branch is not None,
        },
    )


def _list_arg_to_length(arg: Optional[list[str]], num: int) -> Sequence[Optional[str]]:
    if not arg:
        return [None] * num
    if len(arg) == 1:
        return arg * num
    if len(arg) == num:
        return arg
    msg = f"Argument must be of length 1 or {num}"
    raise ValueError(msg)


def parse_dependencies(
    dependencies: Optional[list[str]],
    repo: list[str],
    branch: list[str],
    api_path: list[str],
) -> list[DependencySource]:
    num_deps = max(
        len(dependencies) if dependencies is not None else 0,
        len(repo),
        len(branch),
    )
    if (
        (dependencies and len(dependencies) != num_deps)
        or (api_path and len(api_path) != num_deps)
        or (repo and len(repo) not in [1, num_deps])
        or (branch and len(branch) not in [1, num_deps])
    ):
        msg = (
            "Number of defined repos/branches/api_paths did not match the "
            "number of templates."
        )
        raise ValueError(msg)
    inner_deps = _list_arg_to_length(dependencies, num_deps)
    inner_api_paths = _list_arg_to_length(api_path, num_deps)
    inner_repos = _list_arg_to_length(repo, num_deps)
    inner_branches = _list_arg_to_length(branch, num_deps)

    return [
        parse_dependency_string(iter_dep, iter_repo, iter_branch, iter_api_path)
        for iter_dep, iter_repo, iter_branch, iter_api_path in zip(
            inner_deps,
            inner_repos,
            inner_branches,
            inner_api_paths,
        )
    ]


def _get_repo_path(gitstring: str, ref: Optional[str], repo_dir: Path) -> Path:
    # only based on git for now
    ref_str = ref if ref is not None else ""
    hashed = hashlib.sha256((f"{gitstring}:{ref_str}").encode()).hexdigest()[:8]

    removed_protocol = gitstring.split("://")[-1]
    removed_basename = re.split(r"[/:]", removed_protocol, maxsplit=1)[-1]
    removed_extras = removed_basename.split("#")[0]
    foldername = re.sub(r"\W", "_", removed_extras)

    directory_name = f"{foldername}_{hashed}"
    return repo_dir / directory_name


def update_repo(gitstring: str, ref: Optional[str], repo_dir: Path) -> Path:
    # see if path already saved
    repo_path = _get_repo_path(gitstring, ref, repo_dir)
    if repo_path.exists():
        # try pulling
        try:
            repo = Repo(repo_path)
            if repo.active_branch.name != ref:
                raise ValueError
            repo.remotes.origin.pull()
        except Exception:
            # if it fails, delete and clone again
            shutil.rmtree(repo_path)
            Repo.clone_from(gitstring, repo_path, branch=ref, depth=1)
    else:
        Repo.clone_from(gitstring, repo_path, branch=ref, depth=1)

    return repo_path


def copy_repo(
    source: Path,
    destination: Path,
) -> None:
    """Copiy a repo, ignoring git folders.

    Raises FileNotFound error if it can't find source
    """

    def ignore_func(_, files):
        return [f for f in files if f == ".git"]

    shutil.copytree(source, destination, ignore=ignore_func)
