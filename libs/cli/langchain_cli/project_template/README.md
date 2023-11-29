# LangServeHub Project Template

## Installation

Install the LangChain CLI if you haven't yet

```bash
pip install --user --upgrade git+https://github.com/pingpong-templates/cli.git
```

And install this package's dependencies

```bash
poetry install
```

## Adding packages

```bash
# if you have problems with `poe`, try `poetry run poe`

# adding packages from https://github.com/pingpong-templates/hub
langchain serve add extraction-openai-functions

# adding custom GitHub repo packages
langchain serve add git+https://github.com/hwchase17/chain-of-verification

# with a custom api mount point (defaults to `/{package_name}`)
poe add simple-translator --api_path=/my/custom/path/translator
```

## Removing packages

Note: you remove packages by their api path

```bash
langchain serve remove extraction-openai-functions
```
