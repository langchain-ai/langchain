import os
import re

# Base URL for all class documentation
base_url = "https://api.python.langchain.com/en/latest/"

# Directory where generated markdown files are stored
docs_dir = "docs_skeleton/docs"

# Regular expression to match Python import lines
import_re = re.compile(r"from\s+(\w+(\.\w+)*?)\s+import\s+(\w+)")

def find_files(path):
    """Find all MDX files in the given path"""
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".mdx") or file.endswith(".md"):
                yield os.path.join(root, file)

def replace_imports(file):
    """Replace imports in a markdown file with links to their documentation"""
    with open(file, 'r') as f:
        data = f.read()

    for match in import_re.finditer(data):
        # Create the URL for this class's documentation
        module_path = match.group(1)
        class_name = match.group(3)
        url = base_url + module_path.split(".")[-1] + "/" + module_path.replace(".", "/") + "." + class_name + ".html"

        # Replace the class name with a markdown link
        data = data.replace(match.group(0), f"[{match.group(0)}]({url})")

    with open(file, 'w') as f:
        f.write(data)

def main():
    """Main function"""
    for file in find_files(docs_dir):
        replace_imports(file)

if __name__ == "__main__":
    main()
