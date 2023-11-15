import glob
import os
import re
import shutil
from pathlib import Path

TEMPLATES_DIR = Path(os.path.abspath(__file__)).parents[2] / "templates"
DOCS_TEMPLATES_DIR = Path(os.path.abspath(__file__)).parents[1] / "docs" / "templates"


readmes = list(glob.glob(str(TEMPLATES_DIR) + "/*/README.md"))
destinations = [readme[len(str(TEMPLATES_DIR)) + 1 : -10] + ".md" for readme in readmes]
for source, destination in zip(readmes, destinations):
    full_destination = DOCS_TEMPLATES_DIR / "templates" / destination
    shutil.copyfile(source, full_destination)
    with open(full_destination, "r") as f:
        content = f.read()
    # remove images
    content = re.sub("\!\[.*?\]\((.*?)\)", "", content)
    with open(full_destination, "w") as f:
        f.write(content)

sidebar_hidden = """---
sidebar_class_name: hidden
---

"""

for fn in ("index.md", "contributing.md", "launching_package.md"):
    destination = DOCS_TEMPLATES_DIR / fn
    with open(destination, "r") as f:
        content = f.read()
    # replace relative links
    content = re.sub("\]\(\.\.\/", "](/docs/templates", content)
    content = re.sub("\!\[.*?\]\((.*?)\)", "", content)
    if fn == "index.md":
        content = sidebar_hidden + content
    with open(destination, "w") as f:
        f.write(content)


TEMPLATES_INDEX_DESTINATION = DOCS_TEMPLATES_DIR / "templates" / "index.md"
with open(TEMPLATES_INDEX_DESTINATION, "r") as f:
    content = f.read()
# replace relative links
content = re.sub("\]\(\.\.\/", "](/docs/templates/templates", content)
with open(TEMPLATES_INDEX_DESTINATION, "w") as f:
    f.write(sidebar_hidden + content)
