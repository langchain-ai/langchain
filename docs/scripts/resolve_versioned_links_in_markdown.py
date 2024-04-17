import os
import re
import sys
from pathlib import Path

DOCS_DIR = Path(os.path.abspath(__file__)).parents[1]


def update_links(doc_path, docs_link):
    for path in doc_path.glob('**/*'):
        if path.is_file() and path.suffix in ['.md', '.mdx']:
            with open(path, "r") as f:
                content = f.read()

            # replace relative links
            content = re.sub("\]\(\/docs\/(?!0\.2\.x)", f"]({docs_link}", content)

            with open(path, "w") as f:
                f.write(content)


if __name__ == "__main__":
    update_links(Path(sys.argv[1]), sys.argv[2])