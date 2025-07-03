# TEMPORARY UTILITY SCRIPT: Not for production use. Safe to delete after migration.
import os
import json

# The old and new import lines
OLD_IMPORT = "from langchain.chains.retrieval import create_retrieval_chain"
NEW_IMPORT = "from langchain.chains.retrieval import create_retrieval_chain"

def replace_in_text_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        print(f"Skipped (not UTF-8): {filepath}")
        return
    if OLD_IMPORT in content:
        content = content.replace(OLD_IMPORT, NEW_IMPORT)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Replaced in {filepath}")

def replace_in_ipynb_file(filepath):
    changed = False
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    for cell in data.get("cells", []):
        if cell.get("cell_type") == "code":
            new_lines = []
            for line in cell.get("source", []):
                if OLD_IMPORT in line:
                    line = line.replace(OLD_IMPORT, NEW_IMPORT)
                    changed = True
                new_lines.append(line)
            cell["source"] = new_lines
    if changed:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
        print(f"Replaced in {filepath}")

def main():
    for root, dirs, files in os.walk("."):
        for filename in files:
            filepath = os.path.join(root, filename)
            if filename.endswith(".ipynb"):
                replace_in_ipynb_file(filepath)
            elif filename.endswith((".py", ".md", ".rst", ".txt")):
                replace_in_text_file(filepath)

if __name__ == "__main__":
    main()

