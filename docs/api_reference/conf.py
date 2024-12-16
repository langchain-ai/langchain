"""Configuration file for the Sphinx documentation builder."""

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import json
import os
import sys
from pathlib import Path

import toml
from docutils import nodes
from docutils.parsers.rst.directives.admonitions import BaseAdmonition
from docutils.statemachine import StringList
from sphinx.util.docutils import SphinxDirective

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../../libs/langchain"))

with (_DIR.parents[1] / "libs" / "langchain" / "pyproject.toml").open("r") as f:
    data = toml.load(f)
with (_DIR / "guide_imports.json").open("r") as f:
    imported_classes = json.load(f)


class ExampleLinksDirective(SphinxDirective):
    """Directive to generate a list of links to examples.

    We have a script that extracts links to API reference docs
    from our notebook examples. This directive uses that information
    to backlink to the examples from the API reference docs."""

    has_content = False
    required_arguments = 1

    def run(self):
        """Run the directive.

        Called any time :example_links:`ClassName` is used
        in the template *.rst files."""
        class_or_func_name = self.arguments[0]
        links = imported_classes.get(class_or_func_name, {})
        list_node = nodes.bullet_list()
        for doc_name, link in sorted(links.items()):
            item_node = nodes.list_item()
            para_node = nodes.paragraph()
            link_node = nodes.reference()
            link_node["refuri"] = link
            link_node.append(nodes.Text(doc_name))
            para_node.append(link_node)
            item_node.append(para_node)
            list_node.append(item_node)
        if list_node.children:
            title_node = nodes.rubric()
            title_node.append(nodes.Text(f"Examples using {class_or_func_name}"))
            return [title_node, list_node]
        return [list_node]


class Beta(BaseAdmonition):
    required_arguments = 0
    node_class = nodes.admonition

    def run(self):
        self.content = self.content or StringList(
            [
                (
                    "This feature is in beta. It is actively being worked on, so the "
                    "API may change."
                )
            ]
        )
        self.arguments = self.arguments or ["Beta"]
        return super().run()


def setup(app):
    app.add_directive("example_links", ExampleLinksDirective)
    app.add_directive("beta", Beta)
    app.connect("autodoc-skip-member", skip_private_members)


def skip_private_members(app, what, name, obj, skip, options):
    if skip:
        return True
    if hasattr(obj, "__doc__") and obj.__doc__ and ":private:" in obj.__doc__:
        return True
    if name == "__init__" and obj.__objclass__ is object:
        # dont document default init
        return True
    return None


# -- Project information -----------------------------------------------------

project = "🦜🔗 LangChain"
copyright = "2023, LangChain Inc"
author = "LangChain, Inc"

html_favicon = "_static/img/brand/favicon.png"
html_last_updated_fmt = "%b %d, %Y"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autodoc.typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.autodoc_pydantic",
    "IPython.sphinxext.ipython_console_highlighting",
    "myst_parser",
    "_extensions.gallery_directive",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxcontrib.googleanalytics",
]
source_suffix = [".rst", ".md"]

# some autodoc pydantic options are repeated in the actual template.
# potentially user error, but there may be bugs in the sphinx extension
# with options not being passed through correctly (from either the location in the code)
autodoc_pydantic_model_show_json = False
autodoc_pydantic_field_list_validators = False
autodoc_pydantic_config_members = False
autodoc_pydantic_model_show_config_summary = False
autodoc_pydantic_model_show_validator_members = False
autodoc_pydantic_model_show_validator_summary = False
autodoc_pydantic_model_signature_prefix = "class"
autodoc_pydantic_field_signature_prefix = "param"
autodoc_member_order = "groupwise"
autoclass_content = "both"
autodoc_typehints_format = "short"
autodoc_typehints = "both"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# The theme to use for HTML and HTML Help pages.
html_theme = "pydata_sphinx_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    #     # -- General configuration ------------------------------------------------
    "sidebar_includehidden": True,
    "use_edit_page_button": False,
    #     # "analytics": {
    #     #     "plausible_analytics_domain": "scikit-learn.org",
    #     #     "plausible_analytics_url": "https://views.scientific-python.org/js/script.js",
    #     # },
    #     # If "prev-next" is included in article_footer_items, then setting show_prev_next
    #     # to True would repeat prev and next links. See
    #     # https://github.com/pydata/pydata-sphinx-theme/blob/b731dc230bc26a3d1d1bb039c56c977a9b3d25d8/src/pydata_sphinx_theme/theme/pydata_sphinx_theme/layout.html#L118-L129
    "show_prev_next": False,
    "search_bar_text": "Search",
    "navigation_with_keys": True,
    "collapse_navigation": True,
    "navigation_depth": 3,
    "show_nav_level": 1,
    "show_toc_level": 3,
    "navbar_align": "left",
    "header_links_before_dropdown": 5,
    "header_dropdown_text": "Integrations",
    "logo": {
        "image_light": "_static/wordmark-api.svg",
        "image_dark": "_static/wordmark-api-dark.svg",
    },
    "surface_warnings": True,
    #     # -- Template placement in theme layouts ----------------------------------
    "navbar_start": ["navbar-logo"],
    #     # Note that the alignment of navbar_center is controlled by navbar_align
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["langchain_docs", "theme-switcher", "navbar-icon-links"],
    #     # navbar_persistent is persistent right (even when on mobiles)
    "navbar_persistent": ["search-field"],
    "article_header_start": ["breadcrumbs"],
    "article_header_end": [],
    "article_footer_items": [],
    "content_footer_items": [],
    #     # Use html_sidebars that map page patterns to list of sidebar templates
    #     "primary_sidebar_end": [],
    "footer_start": ["copyright"],
    "footer_center": [],
    "footer_end": [],
    #     # When specified as a dictionary, the keys should follow glob-style patterns, as in
    #     # https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-exclude_patterns
    #     # In particular, "**" specifies the default for all pages
    #     # Use :html_theme.sidebar_secondary.remove: for file-wide removal
    #     "secondary_sidebar_items": {"**": ["page-toc", "sourcelink"]},
    #     "show_version_warning_banner": True,
    #     "announcement": None,
    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            # URL where the link will redirect
            "url": "https://github.com/langchain-ai/langchain",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fa-brands fa-square-github",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        },
        {
            "name": "X / Twitter",
            "url": "https://twitter.com/langchainai",
            "icon": "fab fa-twitter-square",
        },
    ],
    "icon_links_label": "Quick Links",
    "external_links": [],
}


html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "langchain-ai",  # Username
    "github_repo": "langchain",  # Repo name
    "github_version": "master",  # Version
    "conf_py_path": "/docs/api_reference",  # Path in the checkout to the docs root
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# These paths are either relative to html_static_path
# or fully qualified paths (e.g. https://...)
html_css_files = ["css/custom.css"]
html_use_index = False

myst_enable_extensions = ["colon_fence"]

# generate autosummary even if no references
autosummary_generate = True

html_copy_source = False
html_show_sourcelink = False

# Set canonical URL from the Read the Docs Domain
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")

googleanalytics_id = "G-9B66JQQH2F"

# Tell Jinja2 templates the build is running on Read the Docs
if os.environ.get("READTHEDOCS", "") == "True":
    html_context["READTHEDOCS"] = True

master_doc = "index"
