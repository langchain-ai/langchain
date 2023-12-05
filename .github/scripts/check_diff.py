import sys

ALL_DIRS = {
    "libs/core",
    "libs/community",
    "libs/langchain",
    "libs/experimental",
    "libs/partners/openai",
}

if __name__ == "__main__":
    files = sys.argv[1:]
    dirs_to_run = set()

    for file in files:
        if any(dir_ in file for dir_ in (".github/workflows", ".github/tools", ".github/actions", "libs/core")):
            dirs_to_run = ALL_DIRS
            break
        elif "libs/community" in file:
            dirs_to_run.update(("libs/community", "libs/langchain", "libs/experimental"))
        elif "libs/partners" in file:
            partner_dir = file.split("/")[2]
            dirs_to_run.add(f"libs/partners/{partner_dir}")
        else:
            pass
    print(list(dirs_to_run))