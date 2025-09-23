"""Post-process generated HTML files to clean up table-of-contents headers.

Runs after Sphinx generates the API reference HTML. It finds TOC entries like
"ClassName.method_name()" and shortens them to just "method_name()" for better
readability in the sidebar navigation.
"""

import sys
from glob import glob
from pathlib import Path

from bs4 import BeautifulSoup

CUR_DIR = Path(__file__).parents[1]


def process_toc_h3_elements(html_content: str) -> str:
    """Update Class.method() TOC headers to just method()."""
    # Create a BeautifulSoup object
    soup = BeautifulSoup(html_content, "html.parser")

    # Find all <li> elements with class "toc-h3"
    toc_h3_elements = soup.find_all("li", class_="toc-h3")

    # Process each element
    for element in toc_h3_elements:
        try:
            element = element.a.code.span
        except Exception:
            continue
        # Get the text content of the element
        content = element.get_text()

        # Apply the regex substitution
        modified_content = content.split(".")[-1]

        # Update the element's content
        element.string = modified_content

    # Return the modified HTML
    return str(soup)


if __name__ == "__main__":
    dir = sys.argv[1]
    for fn in glob(str(f"{dir.rstrip('/')}/**/*.html"), recursive=True):
        with open(fn, "r") as f:
            html = f.read()
        processed_html = process_toc_h3_elements(html)
        with open(fn, "w") as f:
            f.write(processed_html)
