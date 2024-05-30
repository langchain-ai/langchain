import os
import re
import sys
from pathlib import Path

DOCS_DIR = Path(os.path.abspath(__file__)).parents[1]


def update_links(doc_path, docs_link):
    with open(DOCS_DIR / doc_path, "r") as f:
        content = f.read()

    # replace relative links
    content = re.sub(r"\]\(\.\/", f"]({docs_link}", content)

    frontmatter = """---
custom_edit_url:
---
"""

    with open(DOCS_DIR / doc_path, "w") as f:
        f.write(frontmatter + content)


if __name__ == "__main__":
    update_links(sys.argv[1], sys.argv[2])
