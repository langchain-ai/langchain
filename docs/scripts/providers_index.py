import json
import os
import re
import string
import sys
from collections import defaultdict
from pathlib import Path


def extract_titles(input_dir: str) -> list[dict[str, str]]:
    titles = []
    title_pattern = re.compile(r"^# (.+)")  # Pattern to match '# Title' format

    # Traverse all files in the directory
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)

        if filename.endswith((".md", ".mdx")):
            # Open markdown files and extract title
            with open(file_path, "r", encoding="utf-8") as file:
                for line in file:
                    match = title_pattern.match(line)
                    if match:
                        title = match.group(1)
                        titles.append({"file": filename, "title": title})
                        break  # Stop after the first title line

        elif filename.endswith(".ipynb"):
            # Open Jupyter Notebook files and extract title
            with open(file_path, "r", encoding="utf-8") as file:
                notebook_data = json.load(file)
                # Search in notebook cells for the title
                for cell in notebook_data.get("cells", []):
                    if cell.get("cell_type") == "markdown":
                        for line in cell.get("source", []):
                            match = title_pattern.match(line)
                            if match:
                                title = match.group(1)
                                titles.append({"file": filename, "title": title})
                                break
                    if titles and titles[-1]["file"] == filename:
                        break  # Stop after finding the first title in the notebook

    return titles


def transform_to_links(titles: list[dict[str, str]], prefix: str) -> list[str]:
    return [
        f"[{title['title']}]({prefix}{title['file'].split('.')[0]})" for title in titles
    ]


def generate_index_page(items: list[str], num_columns: int = 5) -> str:
    # Group items by their starting letter (the second character in the string)
    grouped_items = defaultdict(list)
    for item in items:
        first_letter = item[1].upper()
        if first_letter in string.ascii_uppercase:
            grouped_items[first_letter].append(item)
        else:
            grouped_items["0-9"].append(item)  # Non-alphabetical characters go here

    # Sort groups by letters A-Z
    sorted_groups = sorted(grouped_items.items())

    # Generate Markdown content
    content = [
        "# Providers\n\n",
        """
:::info
If you'd like to write your own integration, see [Extending LangChain](/docs/how_to/#custom).

If you'd like to contribute an integration, see [Contributing integrations](/docs/contributing/integrations/).

:::

""",
    ]
    # First part: Menu with links
    menu_links = " | ".join(f"[{letter}](#{letter})" for letter, _ in sorted_groups)
    content.append(menu_links + "\n\n")
    content.append("\n---\n\n")

    # Second part: Grouped items in a single line with separators
    for letter, items in sorted_groups:
        content.append(f"### {letter}\n\n")
        # Sort items within each group and join them in a single line with " | " separator
        items_line = " | ".join(sorted(items))
        content.append(items_line + "\n\n")

    return "".join(content)


if __name__ == "__main__":
    DOCS_DIR = Path(__file__).parents[1]
    input_dir = DOCS_DIR / "docs" / "integrations" / "providers"
    output_dir = DOCS_DIR / Path(sys.argv[1]) / "integrations" / "providers"
    # "index.mdx" is used for `providers` root directory menu
    output_file = output_dir / "all.mdx"

    titles = extract_titles(input_dir)
    links = transform_to_links(titles=titles, prefix="/docs/integrations/providers/")
    mdx_page = generate_index_page(items=links)
    if os.path.isfile(output_file):
        print(f"{output_file} already exists. WE DO NOT overwrite it.")
    else:
        with open(output_file, "w") as f:
            f.write(mdx_page)
            print(f"{output_file} generated successfully")
