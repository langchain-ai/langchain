import json
import sys
import os

LANGCHAIN_DIRS = {
    "libs/core",
    "libs/langchain",
    "libs/experimental",
    "libs/community",
}

if __name__ == "__main__":
    files = sys.argv[1:]
    dirs_to_run = set()

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
                "libs/core",
                ".github/scripts/check_diff.py",
            )
        ):
            dirs_to_run.update(LANGCHAIN_DIRS)
        elif "libs/community" in file:
            dirs_to_run.update(
                ("libs/community", "libs/langchain", "libs/experimental")
            )
        elif "libs/partners" in file:
            partner_dir = file.split("/")[2]
            if os.path.isdir(f"libs/partners/{partner_dir}"):
                dirs_to_run.add(f"libs/partners/{partner_dir}")
            # Skip if the directory was deleted
        elif "libs/langchain" in file:
            dirs_to_run.update(("libs/langchain", "libs/experimental"))
        elif "libs/experimental" in file:
            dirs_to_run.add("libs/experimental")
        elif file.startswith("libs/"):
            dirs_to_run.update(LANGCHAIN_DIRS)
        else:
            pass
    json_output = json.dumps(list(dirs_to_run))
    print(f"dirs-to-run={json_output}")
