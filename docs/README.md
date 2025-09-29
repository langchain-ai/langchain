# LangChain Documentation

For more information on contributing to our documentation, see the [Documentation Contributing Guide](https://python.langchain.com/docs/contributing/how_to/documentation).

## Structure

The primary documentation is located in the `docs/` directory. This directory contains
both the source files for the main documentation as well as the API reference doc
build process.

### API Reference

API reference documentation is located in `docs/api_reference/` and is generated from
the codebase using Sphinx.

The API reference have additional build steps that differ from the main documentation.

#### Deployment Process

Currently, the build process roughly follows these steps:

1. Using the `api_doc_build.yml` GitHub workflow, the API reference docs are
    [built](#build-technical-details) and copied to the `langchain-api-docs-html`
    repository. This workflow is triggered either (1) on a cron routine interval or (2)
    triggered manually.

    In short, the workflow extracts all `langchain-ai`-org-owned repos defined in
    `langchain/libs/packages.yml`, clones them locally (in the workflow runner's file
    system), and then builds the API reference RST files (using `create_api_rst.py`).
    Following post-processing, the HTML files are pushed to the
    `langchain-api-docs-html` repository.
2. After the HTML files are in the `langchain-api-docs-html` repository, they are **not**
    automatically published to the [live docs site](https://python.langchain.com/api_reference/).

    The docs site is served by Vercel. The Vercel deployment process copies the HTML
    files from the `langchain-api-docs-html` repository and deploys them to the live
    site. Deployments are triggered on each new commit pushed to `master`.

#### Build Technical Details

The build process creates a virtual monorepo by syncing multiple repositories, then generates comprehensive API documentation:

1. **Repository Sync Phase:**
   - `.github/scripts/prep_api_docs_build.py` - Clones external partner repos and organizes them into the `libs/partners/` structure to create a virtual monorepo for documentation building

2. **RST Generation Phase:**
   - `docs/api_reference/create_api_rst.py` - Main script that **generates RST files** from Python source code
     - Scans `libs/` directories and extracts classes/functions from each module (using `inspect`)
     - Creates `.rst` files using specialized templates for different object types
     - Templates in `docs/api_reference/templates/` (`pydantic.rst`, `runnable_pydantic.rst`, etc.)

3. **HTML Build Phase:**
   - Sphinx-based, uses `sphinx.ext.autodoc` (auto-extracts docstrings from the codebase)
     - `docs/api_reference/conf.py` (sphinx config) configures `autodoc` and other extensions
   - `sphinx-build` processes the generated `.rst` files into HTML using autodoc
   - `docs/api_reference/scripts/custom_formatter.py` - Post-processes the generated HTML
   - Copies `reference.html` to `index.html` to create the default landing page (artifact? might not need to do this - just put everyhing in index directly?)

4. **Deployment:**
   - `.github/workflows/api_doc_build.yml` - Workflow responsible for orchestrating the entire build and deployment process
   - Built HTML files are committed and pushed to the `langchain-api-docs-html` repository

#### Local Build

For local development and testing of API documentation, use the Makefile targets in the repository root:

```bash
# Full build
make api_docs_build
```

Like the CI process, this target:

- Installs the CLI package in editable mode
- Generates RST files for all packages using `create_api_rst.py`
- Builds HTML documentation with Sphinx
- Post-processes the HTML with `custom_formatter.py`
- Opens the built documentation (`reference.html`) in your browser

**Quick Preview:**

```bash
make api_docs_quick_preview API_PKG=openai
```

- Generates RST files for only the specified package (default: `text-splitters`)
- Builds and post-processes HTML documentation
- Opens the preview in your browser

Both targets automatically clean previous builds and handle the complete build pipeline locally, mirroring the CI process but for faster iteration during development.

#### Documentation Standards

**Docstring Format:**
The API reference uses **Google-style docstrings** with reStructuredText markup. Sphinx processes these through the `sphinx.ext.napoleon` extension to generate documentation.

**Required format:**

```python
def example_function(param1: str, param2: int = 5) -> bool:
    """Brief description of the function.

    Longer description can go here. Use reStructuredText syntax for
    rich formatting like **bold** and *italic*.

    TODO: code: figure out what works?

    Args:
        param1: Description of the first parameter.
        param2: Description of the second parameter with default value.

    Returns:
        Description of the return value.

    Raises:
        ValueError: When param1 is empty.
        TypeError: When param2 is not an integer.

    .. warning::
        This function is experimental and may change.
    """
```

**Special Markers:**

- `:private:` in docstrings excludes members from documentation
- `.. warning::` adds warning admonitions

#### Site Styling and Assets

**Theme and Styling:**

- Uses [**PyData Sphinx Theme**](https://pydata-sphinx-theme.readthedocs.io/en/stable/index.html) (`pydata_sphinx_theme`)
- Custom CSS in `docs/api_reference/_static/css/custom.css` with LangChain-specific:
  - Color palette
  - Inter font family
  - Custom navbar height and sidebar formatting
  - Deprecated/beta feature styling

**Static Assets:**

- Logos: `_static/wordmark-api.svg` (light) and `_static/wordmark-api-dark.svg` (dark mode)
- Favicon: `_static/img/brand/favicon.png`
- Custom CSS: `_static/css/custom.css`

**Post-Processing:**

- `scripts/custom_formatter.py` cleans up generated HTML:
  - Shortens TOC entries from `ClassName.method()` to `method()`

**Analytics and Integration:**

- GitHub integration (source links, edit buttons)
- Example backlinking through custom `ExampleLinksDirective`
