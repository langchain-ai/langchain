import json
import sys

ALL_DIRS = {
    "libs/core",
    "libs/langchain",
    "libs/experimental",
    # "libs/community",
    # "libs/partners/openai",
}

if __name__ == "__main__":
    files = sys.argv[1:]
    dirs_to_run = set()

    for file in files:
        if any(dir_ in file for dir_ in (".github/workflows", ".github/tools", ".github/actions", "libs/core", ".github/scripts/check_diff.py")):
            dirs_to_run = ALL_DIRS
            break
        elif "libs/community" in file:
            dirs_to_run.update(("libs/community", "libs/langchain", "libs/experimental"))
        elif "libs/partners" in file:
            partner_dir = file.split("/")[2]
            dirs_to_run.update((f"libs/partners/{partner_dir}", "libs/langchain", "libs/experimental"))
        elif "libs/langchain" in file:
            dirs_to_run.update(("libs/langchain", "libs/experimental"))
        elif "libs/experimental" in file:
            dirs_to_run.add("libs/experimental")
        else:
            pass
    print(json.dumps(list(dirs_to_run)))