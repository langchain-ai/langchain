from typing import Optional, TypedDict
from pathlib import Path

import shutil
import re
from langchain_cli.constants import DEFAULT_GIT_REPO, DEFAULT_GIT_SUBDIRECTORY
import hashlib
from git import Repo


class DependencySource(TypedDict):
    git: str
    ref: Optional[str]
    subdirectory: Optional[str]


def _get_main_branch(repo: Repo) -> Optional[str]:
    """
    Get the name of the main branch of a git repo.
    From https://stackoverflow.com/questions/69651536/how-to-get-master-main-branch-from-gitpython
    """
    try:
        # replace "origin" with your remote name if differs
        show_result = repo.git.remote("show", "origin")

        # The show_result contains a wall of text in the language that
        # is set by your locales. Now you can use regex to extract the
        # default branch name, but if your language is different
        # from english, you need to adjust this regex pattern.

        matches = re.search(r"\s*HEAD branch:\s*(.*)", show_result)
        if matches:
            default_branch = matches.group(1)
            return default_branch
    except Exception:
        pass
    # fallback to main/master
    if "main" in repo.heads:
        return "main"
    if "master" in repo.heads:
        return "master"

    raise ValueError("Could not find main branch")


# use poetry dependency string format
def _parse_dependency_string(package_string: str) -> DependencySource:
    if package_string.startswith("git+"):
        # remove git+
        remaining = package_string[4:]
        # split main string from params
        gitstring, *params = remaining.split("#")
        # parse params
        params_dict = {}
        for param in params:
            if not param:
                # ignore empty entries
                continue
            if "=" in param:
                key, value = param.split("=")
                if key in params_dict:
                    raise ValueError(
                        f"Duplicate parameter {key} in dependency string {package_string}"
                    )
                params_dict[key] = value
            else:
                if "ref" in params_dict:
                    raise ValueError(
                        f"Duplicate parameter ref in dependency string {package_string}"
                    )
                params_dict["ref"] = param
        return DependencySource(
            git=gitstring,
            ref=params_dict.get("ref"),
            subdirectory=params_dict.get("subdirectory"),
        )

    elif package_string.startswith("https://"):
        raise NotImplementedError("url dependencies are not supported yet")
    else:
        # it's a default git repo dependency
        gitstring = DEFAULT_GIT_REPO
        subdirectory = str(Path(DEFAULT_GIT_SUBDIRECTORY) / package_string)
        return DependencySource(git=gitstring, ref=None, subdirectory=subdirectory)


def _get_repo_path(dependency: DependencySource, repo_dir: Path) -> Path:
    # only based on git for now
    gitstring = dependency["git"]
    hashed = hashlib.sha256(gitstring.encode("utf-8")).hexdigest()[:8]

    removed_protocol = gitstring.split("://")[-1]
    removed_basename = re.split(r"[/:]", removed_protocol, 1)[-1]
    removed_extras = removed_basename.split("#")[0]
    foldername = re.sub(r"[^a-zA-Z0-9_]", "_", removed_extras)

    directory_name = f"{foldername}_{hashed}"
    return repo_dir / directory_name


def update_repo(gitpath: str, repo_dir: Path) -> Path:
    # see if path already saved
    dependency = _parse_dependency_string(gitpath)
    repo_path = _get_repo_path(dependency, repo_dir)
    if not repo_path.exists():
        repo = Repo.clone_from(dependency["git"], repo_path)
    else:
        repo = Repo(repo_path)

    # pull it
    ref = dependency.get("ref") if dependency.get("ref") else _get_main_branch(repo)
    repo.git.checkout(ref)

    repo.git.pull()

    return (
        repo_path
        if dependency["subdirectory"] is None
        else repo_path / dependency["subdirectory"]
    )


def copy_repo(
    source: Path,
    destination: Path,
) -> None:
    def ignore_func(_, files):
        return [f for f in files if f == ".git"]

    shutil.copytree(source, destination, ignore=ignore_func)
