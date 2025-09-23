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

#### Build Process

Currently, the build process roughly follows these steps:

1. Using the `api_doc_build.yml` GitHub workflow, the API reference docs are built and
    copied to the `langchain-api-docs-html` repository. This workflow is triggered
    either (1) on a cron routine interval or (2) triggered manually.

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

#### Technical Details

- Sphinx-based, uses `sphinx.ext.autodoc` (auto-extracts docstrings from the codebase)
  - [`conf.py`](http://conf.py) (sphinx config) configures `autodoc` and other extensions
- `create_api_rst.py` - Main script that **generates RST files** from Python source code
  - Scans `libs/` directories and extracts classes/functions from each module (using `inspect`)
  - Creates `.rst` files using template for object type
    - Templates in `docs/api_reference/templates`
  - `sphinx-build` processes `.rst` to HTML using autodoc
- `api_doc_build.yml` - workflow responsible for building and deploying the API ref
  - `prep_api_docs_build.py` - sync libraries into the virtual monorepo, placing them in
      the partners structure
