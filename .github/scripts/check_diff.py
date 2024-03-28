import json
import sys
import os
from typing import Dict

LANGCHAIN_DIRS = [
    "libs/core",
    "libs/text-splitters",
    "libs/community",
    "libs/langchain",
    "libs/experimental",
]


def _is_not_partner_tombstone(dir_):
    return os.path.isdir(dir_) and [
        filename for filename in os.listdir(dir_) if not filename.startswith(".")
    ] != ["README.md"]


PARTNER_DIRS = [
    f"libs/partners/{d}"
    for d in os.listdir("libs/partners")
    if _is_not_partner_tombstone(f"libs/partners/{d}")
]


if __name__ == "__main__":
    files = sys.argv[1:]

    dirs_to_run: Dict[str, set] = {
        "lint": set(),
        "test": set(),
        "extended-test": set(),
    }

    if len(files) == 300:
        # max diff length is 300 files - there are likely files missing
        raise ValueError("Max diff reached. Please manually run CI on changed libs.")

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
                if file.startswith(dir_):
                    found = True
                if found:
                    dirs_to_run["extended-test"].add(dir_)
            if file.startswith("libs/core"):
                dirs_to_run["test"].update(PARTNER_DIRS)
        elif file.startswith("libs/cli"):
            # todo: add cli makefile
            pass
        elif file.startswith("libs/partners"):
            partner_dir = file.split("/")[2]
            partner_path = f"libs/partners/{partner_dir}"
            if _is_not_partner_tombstone(partner_path):
                dirs_to_run["test"].add(partner_path)
            # Skip if the directory was deleted or is just a tombstone readme
        elif file.startswith("libs/"):
            raise ValueError(
                f"Unknown lib: {file}. check_diff.py likely needs "
                "an update for this new library!"
            )
        elif any(file.startswith(p) for p in ["docs/", "templates/", "cookbook/"]):
            dirs_to_run["lint"].add(".")

    outputs = {
        "dirs-to-lint": list(
            dirs_to_run["lint"] | dirs_to_run["test"] | dirs_to_run["extended-test"]
        ),
        "dirs-to-test": list(dirs_to_run["test"] | dirs_to_run["extended-test"]),
        "dirs-to-extended-test": list(dirs_to_run["extended-test"]),
    }
    for key, value in outputs.items():
        json_output = json.dumps(value)
        print(f"{key}={json_output}")  # noqa: T201
