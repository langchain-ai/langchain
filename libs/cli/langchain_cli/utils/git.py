import hashlib
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, TypedDict

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
    event_metadata: Dict


# use poetry dependency string format
def parse_dependency_string(
    dep: Optional[str],
    repo: Optional[str],
    branch: Optional[str],
    api_path: Optional[str],
) -> DependencySource:
    if dep is not None and dep.startswith("git+"):
        if repo is not None or branch is not None:
            raise ValueError(
                "If a dependency starts with git+, you cannot manually specify "
                "a repo or branch."
            )
        # remove git+
        gitstring = dep[4:]
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
            api_path=api_path,
            event_metadata={"dependency_string": dep},
        )

    elif dep is not None and dep.startswith("https://"):
        raise ValueError("Only git dependencies are supported")
    else:
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


def _list_arg_to_length(arg: Optional[List[str]], num: int) -> Sequence[Optional[str]]:
    if not arg:
        return [None] * num
    elif len(arg) == 1:
        return arg * num
    elif len(arg) == num:
        return arg
    else:
        raise ValueError(f"Argument must be of length 1 or {num}")


def parse_dependencies(
    dependencies: Optional[List[str]],
    repo: List[str],
    branch: List[str],
    api_path: List[str],
) -> List[DependencySource]:
    num_deps = max(
        len(dependencies) if dependencies is not None else 0, len(repo), len(branch)
    )
    if (
        (dependencies and len(dependencies) != num_deps)
        or (api_path and len(api_path) != num_deps)
        or (repo and len(repo) not in [1, num_deps])
        or (branch and len(branch) not in [1, num_deps])
    ):
        raise ValueError(
            "Number of defined repos/branches/api_paths did not match the "
            "number of templates."
        )
    inner_deps = _list_arg_to_length(dependencies, num_deps)
    inner_api_paths = _list_arg_to_length(api_path, num_deps)
    inner_repos = _list_arg_to_length(repo, num_deps)
    inner_branches = _list_arg_to_length(branch, num_deps)

    return [
        parse_dependency_string(iter_dep, iter_repo, iter_branch, iter_api_path)
        for iter_dep, iter_repo, iter_branch, iter_api_path in zip(
            inner_deps, inner_repos, inner_branches, inner_api_paths
        )
    ]


def _get_repo_path(gitstring: str, ref: Optional[str], repo_dir: Path) -> Path:
    # only based on git for now
    ref_str = ref if ref is not None else ""
    hashed = hashlib.sha256((f"{gitstring}:{ref_str}").encode("utf-8")).hexdigest()[:8]

    removed_protocol = gitstring.split("://")[-1]
    removed_basename = re.split(r"[/:]", removed_protocol, 1)[-1]
    removed_extras = removed_basename.split("#")[0]
    foldername = re.sub(r"[^a-zA-Z0-9_]", "_", removed_extras)

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
                raise ValueError()
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
    """
    Copies a repo, ignoring git folders.

    Raises FileNotFound error if it can't find source
    """

    def ignore_func(_, files):
        return [f for f in files if f == ".git"]

    shutil.copytree(source, destination, ignore=ignore_func)
