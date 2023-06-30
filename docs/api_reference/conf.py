"""Configuration file for the Sphinx documentation builder."""
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

import toml

sys.path.insert(0, os.path.abspath("."))

with open("../../pyproject.toml") as f:
    data = toml.load(f)

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

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "inherited-members": "BaseModel",
    "undoc-members": True,
    "special-members": "__call__",
}
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
# or fully qualified paths (eg. https://...)
html_css_files = [
    "css/custom.css",
]
html_use_index = False

myst_enable_extensions = ["colon_fence"]

# generate autosummary even if no references
autosummary_generate = True
