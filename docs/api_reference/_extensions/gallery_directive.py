"""A directive to generate a gallery of images from structured data.

Generating a gallery of images that are all the same size is a common
pattern in documentation, and this can be cumbersome if the gallery is
generated programmatically. This directive wraps this particular use-case
in a helper-directive to generate it with a single YAML configuration file.

It currently exists for maintainers of the pydata-sphinx-theme,
but might be abstracted into a standalone package if it proves useful.
"""

from pathlib import Path
from typing import Any, ClassVar, Dict, List

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.application import Sphinx
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective
from yaml import safe_load

logger = logging.getLogger(__name__)


TEMPLATE_GRID = """
`````{{grid}} {columns}
{options}

{content}

`````
"""

GRID_CARD = """
````{{grid-item-card}} {title}
{options}

{content}
````
"""


class GalleryGridDirective(SphinxDirective):
    """A directive to show a gallery of images and links in a Bootstrap grid.

    The grid can be generated from a YAML file that contains a list of items, or
    from the content of the directive (also formatted in YAML). Use the parameter
    "class-card" to add an additional CSS class to all cards. When specifying the grid
    items, you can use all parameters from "grid-item-card" directive to customize
    individual cards + ["image", "header", "content", "title"].

    Danger:
        This directive can only be used in the context of a Myst documentation page as
        the templates use Markdown flavored formatting.
    """

    name = "gallery-grid"
    has_content = True
    required_arguments = 0
    optional_arguments = 1
    final_argument_whitespace = True
    option_spec: ClassVar[dict[str, Any]] = {
        # A class to be added to the resulting container
        "grid-columns": directives.unchanged,
        "class-container": directives.unchanged,
        "class-card": directives.unchanged,
    }

    def run(self) -> List[nodes.Node]:
        """Create the gallery grid."""
        if self.arguments:
            # If an argument is given, assume it's a path to a YAML file
            # Parse it and load it into the directive content
            path_data_rel = Path(self.arguments[0])
            path_doc, _ = self.get_source_info()
            path_doc = Path(path_doc).parent
            path_data = (path_doc / path_data_rel).resolve()
            if not path_data.exists():
                logger.info(f"Could not find grid data at {path_data}.")
                nodes.text("No grid data found at {path_data}.")
                return
            yaml_string = path_data.read_text()
        else:
            yaml_string = "\n".join(self.content)

        # Use all the element with an img-bottom key as sites to show
        # and generate a card item for each of them
        grid_items = []
        for item in safe_load(yaml_string):
            # remove parameters that are not needed for the card options
            title = item.pop("title", "")

            # build the content of the card using some extra parameters
            header = f"{item.pop('header')}  \n^^^  \n" if "header" in item else ""
            image = f"![image]({item.pop('image')})  \n" if "image" in item else ""
            content = f"{item.pop('content')}  \n" if "content" in item else ""

            # optional parameter that influence all cards
            if "class-card" in self.options:
                item["class-card"] = self.options["class-card"]

            loc_options_str = "\n".join(f":{k}: {v}" for k, v in item.items()) + "  \n"

            card = GRID_CARD.format(
                options=loc_options_str, content=header + image + content, title=title
            )
            grid_items.append(card)

        # Parse the template with Sphinx Design to create an output container
        # Prep the options for the template grid
        class_ = "gallery-directive" + f' {self.options.get("class-container", "")}'
        options = {"gutter": 2, "class-container": class_}
        options_str = "\n".join(f":{k}: {v}" for k, v in options.items())

        # Create the directive string for the grid
        grid_directive = TEMPLATE_GRID.format(
            columns=self.options.get("grid-columns", "1 2 3 4"),
            options=options_str,
            content="\n".join(grid_items),
        )

        # Parse content as a directive so Sphinx Design processes it
        container = nodes.container()
        self.state.nested_parse([grid_directive], 0, container)

        # Sphinx Design outputs a container too, so just use that
        return [container.children[0]]


def setup(app: Sphinx) -> Dict[str, Any]:
    """Add custom configuration to sphinx app.

    Args:
        app: the Sphinx application

    Returns:
        the 2 parallel parameters set to ``True``.
    """
    app.add_directive("gallery-grid", GalleryGridDirective)

    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
