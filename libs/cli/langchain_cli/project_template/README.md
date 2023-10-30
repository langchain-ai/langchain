# LangServe App Template

## Installation

Install the LangChain CLI if you haven't yet

```bash
pip install --upgrade "langchain-cli[serve]"
```

## Adding packages

```bash
# if you have problems with `poe`, try `poetry run poe`

# adding packages from 
# https://github.com/langchain-ai/langchain/tree/master/templates
langchain app add extraction-openai-functions

# adding custom GitHub repo packages
langchain app add --repo hwchase17/chain-of-verification
# or with whole git string (supports other git providers):
# langchain app add git+https://github.com/hwchase17/chain-of-verification

# with a custom api mount point (defaults to `/{package_name}`)
langchain app add rag-chroma --api_path=/my/custom/path/rag
```

## Removing packages

Note: you remove packages by their api path

```bash
langchain app remove my/custom/path/rag
```
