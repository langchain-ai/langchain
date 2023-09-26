import os

import py
import pytest

from langchain.document_loaders import GitLoader


def init_repo(tmpdir: py.path.local, dir_name: str) -> str:
    from git import Repo

    repo_dir = tmpdir.mkdir(dir_name)
    repo = Repo.init(repo_dir)
    git = repo.git
    git.checkout(b="main")

    git.config("user.name", "Test User")
    git.config("user.email", "test@example.com")

    sample_file = "file.txt"
    with open(os.path.join(repo_dir, sample_file), "w") as f:
        f.write("content")
    git.add([sample_file])
    git.commit(m="Initial commit")

    return str(repo_dir)


@pytest.mark.requires("git")
def test_load_twice(tmpdir: py.path.local) -> None:
    """
    Test that loading documents twice from the same repository does not raise an error.
    """

    clone_url = init_repo(tmpdir, "remote_repo")

    repo_path = tmpdir.mkdir("local_repo").strpath
    loader = GitLoader(repo_path=repo_path, clone_url=clone_url)

    documents = loader.load()
    assert len(documents) == 1

    documents = loader.load()
    assert len(documents) == 1


@pytest.mark.requires("git")
def test_clone_different_repo(tmpdir: py.path.local) -> None:
    """
    Test that trying to clone a different repository into a directory already
    containing a clone raises a ValueError.
    """

    clone_url = init_repo(tmpdir, "remote_repo")

    repo_path = tmpdir.mkdir("local_repo").strpath
    loader = GitLoader(repo_path=repo_path, clone_url=clone_url)

    documents = loader.load()
    assert len(documents) == 1

    other_clone_url = init_repo(tmpdir, "other_remote_repo")
    other_loader = GitLoader(repo_path=repo_path, clone_url=other_clone_url)
    with pytest.raises(ValueError):
        other_loader.load()
