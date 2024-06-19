import glob
import os
import re
import shutil
import sys
from pathlib import Path

if __name__ == "__main__":
    intermediate_dir = Path(sys.argv[1])

    templates_source_dir = Path(os.path.abspath(__file__)).parents[2] / "templates"
    templates_intermediate_dir = intermediate_dir / "templates"

    readmes = list(glob.glob(str(templates_source_dir) + "/*/README.md"))
    destinations = [
        readme[len(str(templates_source_dir)) + 1 : -10] + ".md" for readme in readmes
    ]
    for source, destination in zip(readmes, destinations):
        full_destination = templates_intermediate_dir / destination
        shutil.copyfile(source, full_destination)
        with open(full_destination, "r") as f:
            content = f.read()
        # remove images
        content = re.sub(r"\!\[.*?\]\((.*?)\)", "", content)
        with open(full_destination, "w") as f:
            f.write(content)

    sidebar_hidden = """---
sidebar_class_name: hidden
custom_edit_url:
---

"""

    # handle index file
    templates_index_source = templates_source_dir / "docs" / "INDEX.md"
    templates_index_intermediate = templates_intermediate_dir / "index.md"

    with open(templates_index_source, "r") as f:
        content = f.read()

    # replace relative links
    content = re.sub(r"\]\(\.\.\/", "](/docs/templates/", content)

    with open(templates_index_intermediate, "w") as f:
        f.write(sidebar_hidden + content)
