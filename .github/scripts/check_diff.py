import glob
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Set
from pathlib import Path
import tomllib

from packaging.requirements import Requirement

from get_min_versions import get_min_version_from_toml


LANGCHAIN_DIRS = [
    "libs/core",
    "libs/text-splitters",
    "libs/langchain",
    "libs/community",
]

# when set to True, we are ignoring core dependents
# in order to be able to get CI to pass for each individual
# package that depends on core
# e.g. if you touch core, we don't then add textsplitters/etc to CI
IGNORE_CORE_DEPENDENTS = False

# ignored partners are removed from dependents
# but still run if directly edited
IGNORED_PARTNERS = [
    # remove huggingface from dependents because of CI instability
    # specifically in huggingface jobs
    # https://github.com/langchain-ai/langchain/issues/25558
    "huggingface",
    # prompty exhibiting issues with numpy for Python 3.13
    # https://github.com/langchain-ai/langchain/actions/runs/12651104685/job/35251034969?pr=29065
    "prompty",
]

PY_312_MAX_PACKAGES = [
    "libs/partners/huggingface",  # https://github.com/pytorch/pytorch/issues/130249
    "libs/partners/voyageai",
]


def all_package_dirs() -> Set[str]:
    return {
        "/".join(path.split("/")[:-1]).lstrip("./")
        for path in glob.glob("./libs/**/pyproject.toml", recursive=True)
        if "libs/cli" not in path and "libs/standard-tests" not in path
    }


def dependents_graph() -> dict:
    """
    Construct a mapping of package -> dependents, such that we can
    run tests on all dependents of a package when a change is made.
    """
    dependents = defaultdict(set)

    for path in glob.glob("./libs/**/pyproject.toml", recursive=True):
        if "template" in path:
            continue

        # load regular and test deps from pyproject.toml
        with open(path, "rb") as f:
            pyproject = tomllib.load(f)

        pkg_dir = "libs" + "/".join(path.split("libs")[1].split("/")[:-1])
        for dep in [
            *pyproject["project"]["dependencies"],
            *pyproject["dependency-groups"]["test"],
        ]:
            requirement = Requirement(dep)
            package_name = requirement.name
            if "langchain" in dep:
                dependents[package_name].add(pkg_dir)
                continue

        # load extended deps from extended_testing_deps.txt
        package_path = Path(path).parent
        extended_requirement_path = package_path / "extended_testing_deps.txt"
        if extended_requirement_path.exists():
            with open(extended_requirement_path, "r") as f:
                extended_deps = f.read().splitlines()
                for depline in extended_deps:
                    if depline.startswith("-e "):
                        # editable dependency
                        assert depline.startswith(
                            "-e ../partners/"
                        ), "Extended test deps should only editable install partner packages"
                        partner = depline.split("partners/")[1]
                        dep = f"langchain-{partner}"
                    else:
                        dep = depline.split("==")[0]

                    if "langchain" in dep:
                        dependents[dep].add(pkg_dir)

    for k in dependents:
        for partner in IGNORED_PARTNERS:
            if f"libs/partners/{partner}" in dependents[k]:
                dependents[k].remove(f"libs/partners/{partner}")
    return dependents


def add_dependents(dirs_to_eval: Set[str], dependents: dict) -> List[str]:
    updated = set()
    for dir_ in dirs_to_eval:
        # handle core manually because it has so many dependents
        if "core" in dir_:
            updated.add(dir_)
            continue
        pkg = "langchain-" + dir_.split("/")[-1]
        updated.update(dependents[pkg])
        updated.add(dir_)
    return list(updated)


def _get_configs_for_single_dir(job: str, dir_: str) -> List[Dict[str, str]]:
    if job == "test-pydantic":
        return _get_pydantic_test_configs(dir_)

    if dir_ == "libs/core":
        py_versions = ["3.9", "3.10", "3.11", "3.12", "3.13"]
    # custom logic for specific directories
    elif dir_ == "libs/partners/milvus":
        # milvus doesn't allow 3.12 because they declare deps in funny way
        py_versions = ["3.9", "3.11"]

    elif dir_ in PY_312_MAX_PACKAGES:
        py_versions = ["3.9", "3.12"]

    elif dir_ == "libs/langchain" and job == "extended-tests":
        py_versions = ["3.9", "3.13"]

    elif dir_ == "libs/community" and job == "extended-tests":
        py_versions = ["3.9", "3.12"]

    elif dir_ == "libs/community" and job == "compile-integration-tests":
        # community integration deps are slow in 3.12
        py_versions = ["3.9", "3.11"]
    elif dir_ == ".":
        # unable to install with 3.13 because tokenizers doesn't support 3.13 yet
        py_versions = ["3.9", "3.12"]
    else:
        py_versions = ["3.9", "3.13"]

    return [{"working-directory": dir_, "python-version": py_v} for py_v in py_versions]


def _get_pydantic_test_configs(
    dir_: str, *, python_version: str = "3.11"
) -> List[Dict[str, str]]:
    with open("./libs/core/uv.lock", "rb") as f:
        core_uv_lock_data = tomllib.load(f)
    for package in core_uv_lock_data["package"]:
        if package["name"] == "pydantic":
            core_max_pydantic_minor = package["version"].split(".")[1]
            break

    with open(f"./{dir_}/uv.lock", "rb") as f:
        dir_uv_lock_data = tomllib.load(f)

    for package in dir_uv_lock_data["package"]:
        if package["name"] == "pydantic":
            dir_max_pydantic_minor = package["version"].split(".")[1]
            break

    core_min_pydantic_version = get_min_version_from_toml(
        "./libs/core/pyproject.toml", "release", python_version, include=["pydantic"]
    )["pydantic"]
    core_min_pydantic_minor = (
        core_min_pydantic_version.split(".")[1]
        if "." in core_min_pydantic_version
        else "0"
    )
    dir_min_pydantic_version = get_min_version_from_toml(
        f"./{dir_}/pyproject.toml", "release", python_version, include=["pydantic"]
    ).get("pydantic", "0.0.0")
    dir_min_pydantic_minor = (
        dir_min_pydantic_version.split(".")[1]
        if "." in dir_min_pydantic_version
        else "0"
    )

    custom_mins = {
        # depends on pydantic-settings 2.4 which requires pydantic 2.7
        "libs/community": 7,
    }

    max_pydantic_minor = min(
        int(dir_max_pydantic_minor),
        int(core_max_pydantic_minor),
    )
    min_pydantic_minor = max(
        int(dir_min_pydantic_minor),
        int(core_min_pydantic_minor),
        custom_mins.get(dir_, 0),
    )

    configs = [
        {
            "working-directory": dir_,
            "pydantic-version": f"2.{v}.0",
            "python-version": python_version,
        }
        for v in range(min_pydantic_minor, max_pydantic_minor + 1)
    ]
    return configs


def _get_configs_for_multi_dirs(
    job: str, dirs_to_run: Dict[str, Set[str]], dependents: dict
) -> List[Dict[str, str]]:
    if job == "lint":
        dirs = add_dependents(
            dirs_to_run["lint"] | dirs_to_run["test"] | dirs_to_run["extended-test"],
            dependents,
        )
    elif job in ["test", "compile-integration-tests", "dependencies", "test-pydantic"]:
        dirs = add_dependents(
            dirs_to_run["test"] | dirs_to_run["extended-test"], dependents
        )
    elif job == "extended-tests":
        dirs = list(dirs_to_run["extended-test"])
    else:
        raise ValueError(f"Unknown job: {job}")

    return [
        config for dir_ in dirs for config in _get_configs_for_single_dir(job, dir_)
    ]


if __name__ == "__main__":
    files = sys.argv[1:]

    dirs_to_run: Dict[str, set] = {
        "lint": set(),
        "test": set(),
        "extended-test": set(),
    }
    docs_edited = False

    if len(files) >= 300:
        # max diff length is 300 files - there are likely files missing
        dirs_to_run["lint"] = all_package_dirs()
        dirs_to_run["test"] = all_package_dirs()
        dirs_to_run["extended-test"] = set(LANGCHAIN_DIRS)

    for file in files:
        if any(
            file.startswith(dir_)
            for dir_ in (
                ".github/workflows",
                ".github/tools",
                ".github/actions",
                ".github/scripts/check_diff.py",
            )
        ):
            # add all LANGCHAIN_DIRS for infra changes
            dirs_to_run["extended-test"].update(LANGCHAIN_DIRS)
            dirs_to_run["lint"].add(".")

        if any(file.startswith(dir_) for dir_ in LANGCHAIN_DIRS):
            # add that dir and all dirs after in LANGCHAIN_DIRS
            # for extended testing

            found = False
            for dir_ in LANGCHAIN_DIRS:
                if dir_ == "libs/core" and IGNORE_CORE_DEPENDENTS:
                    dirs_to_run["extended-test"].add(dir_)
                    continue
                if file.startswith(dir_):
                    found = True
                if found:
                    dirs_to_run["extended-test"].add(dir_)
        elif file.startswith("libs/standard-tests"):
            # TODO: update to include all packages that rely on standard-tests (all partner packages)
            # note: won't run on external repo partners
            dirs_to_run["lint"].add("libs/standard-tests")
            dirs_to_run["test"].add("libs/standard-tests")
            dirs_to_run["lint"].add("libs/cli")
            dirs_to_run["test"].add("libs/cli")
            dirs_to_run["test"].add("libs/partners/mistralai")
            dirs_to_run["test"].add("libs/partners/openai")
            dirs_to_run["test"].add("libs/partners/anthropic")
            dirs_to_run["test"].add("libs/partners/fireworks")
            dirs_to_run["test"].add("libs/partners/groq")

        elif file.startswith("libs/cli"):
            dirs_to_run["lint"].add("libs/cli")
            dirs_to_run["test"].add("libs/cli")
            
        elif file.startswith("libs/partners"):
            partner_dir = file.split("/")[2]
            if os.path.isdir(f"libs/partners/{partner_dir}") and [
                filename
                for filename in os.listdir(f"libs/partners/{partner_dir}")
                if not filename.startswith(".")
            ] != ["README.md"]:
                dirs_to_run["test"].add(f"libs/partners/{partner_dir}")
            # Skip if the directory was deleted or is just a tombstone readme
        elif file == "libs/packages.yml":
            continue
        elif file.startswith("libs/"):
            raise ValueError(
                f"Unknown lib: {file}. check_diff.py likely needs "
                "an update for this new library!"
            )
        elif file.startswith("docs/") or file in ["pyproject.toml", "uv.lock"]: # docs or root uv files
            docs_edited = True
            dirs_to_run["lint"].add(".")

    dependents = dependents_graph()

    # we now have dirs_by_job
    # todo: clean this up
    map_job_to_configs = {
        job: _get_configs_for_multi_dirs(job, dirs_to_run, dependents)
        for job in [
            "lint",
            "test",
            "extended-tests",
            "compile-integration-tests",
            "dependencies",
            "test-pydantic",
        ]
    }
    map_job_to_configs["test-doc-imports"] = (
        [{"python-version": "3.12"}] if docs_edited else []
    )

    for key, value in map_job_to_configs.items():
        json_output = json.dumps(value)
        print(f"{key}={json_output}")
