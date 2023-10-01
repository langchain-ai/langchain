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
from sphinx.util.docutils import SphinxDirective

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../../libs/langchain"))
sys.path.insert(0, os.path.abspath("../../libs/experimental"))

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
        for doc_name, link in links.items():
            item_node = nodes.list_item()
            para_node = nodes.paragraph()
            link_node = nodes.reference()
            link_node["refuri"] = link
            link_node.append(nodes.Text(doc_name))
            para_node.append(link_node)
            item_node.append(para_node)
            list_node.append(item_node)
        if list_node.children:
            title_node = nodes.title()
            title_node.append(nodes.Text(f"Examples using {class_or_func_name}"))
            return [title_node, list_node]
        return [list_node]


def setup(app):
    app.add_directive("example_links", ExampleLinksDirective)


# -- Project information -----------------------------------------------------

project = "🦜🔗 LangChain"
copyright = "2023, Harrison Chase"
author = "Harrison Chase"

version = data["tool"]["poetry"]["version"]
release = version

html_title = project + " " + version
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
    "sphinx_copybutton",
    "sphinx_panels",
    "IPython.sphinxext.ipython_console_highlighting",
]
source_suffix = [".rst"]

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

# autodoc_typehints = "description"
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
html_theme = "scikit-learn-modern"
html_theme_path = ["themes"]

# redirects dictionary maps from old links to new links
html_additional_pages = {}
redirects = {
    "index": "api_reference",
}
for old_link in redirects:
    html_additional_pages[old_link] = "redirects.html"

html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "hwchase17",  # Username
    "github_repo": "langchain",  # Repo name
    "github_version": "master",  # Version
    "conf_py_path": "/docs/api_reference",  # Path in the checkout to the docs root
    "redirects": redirects,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# These paths are either relative to html_static_path
# or fully qualified paths (e.g. https://...)
html_css_files = [
    "css/custom.css",
]
html_use_index = False

myst_enable_extensions = ["colon_fence"]

# generate autosummary even if no references
autosummary_generate = True
