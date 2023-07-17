import os
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory where generated markdown files are stored
docs_dir = "docs_skeleton/docs"

# Regular expression to match Python import lines
link_re = re.compile(r"\[(from langchain.+?)\]\(.+?\)")


def find_files(path):
    """Find all MDX files in the given path"""
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".mdx") or file.endswith(".md"):
                yield os.path.join(root, file)


def unreplace_imports(file):
    """Replace import links in a markdown file with just the import text"""
    with open(file, "r") as f:
        data = f.read()

    for match in link_re.finditer(data):
        # Replace the markdown link with just the link text
        data = data.replace(match.group(0), match.group(1))

    with open(file, "w") as f:
        f.write(data)


def main():
    """Main function"""
    for file in find_files(docs_dir):
        print(f"Removing links from imports in {file}")
        unreplace_imports(file)


if __name__ == "__main__":
    main()
