# LangServe Hub

Packages that can be easily hosted by LangServe using the `langserve` cli.

## Using LangServe Hub

You can install the `langservehub` CLI and use it as follows:
```bash
# install langservehub CLI
pip install --upgrade langservehub

langservehub new my-app
cd my-app

poetry install

# if you have problems with poe, use `poetry run poe ...` instead

# add the simple-pirate package
poe add --repo=pingpong-templates/hub simple-pirate

# adding other GitHub repo packages, defaults to repo root
poe add --repo=hwchase17/chain-of-verification

# with a custom api mount point (defaults to `/{package_name}`)
poe add --repo=pingpong-templates/hub simple-translator --api_path=/my/custom/path/translator

poe list

poe start
^C

# remove packages by their api path:
poe remove my/custom/path/translator
```

## Creating New Packages

You can also create new packages with the `langservehub package new` command

```bash
# starting from this directory in langserve-hub
langservehub package new simple-newpackage
```

Now you can edit the chain in `simple-newpackage/simple_newpackage/chain.py` and put up a PR!

Your package will be usable as `poe add --repo=pingpong-templates/hub simple-newpackage` when it's merged in.

## Data Format

What makes these packages work?

- Poetry
- pyproject.toml files

### Installable Packages

Everything is a Poetry package currently. This allows poetry to manage our dependencies for us :).

In addition to normal keys in the `pyproject.toml` file, you'll notice an additional `tool.langserve` key ([link](https://github.com/langchain-ai/langserve-hub/blob/main/simple/pirate/pyproject.toml#L13-L15)).

This allows us to identify which module and attribute to import as the chain/runnable for the langserve `add_routes` call.

### Apps (with installed langserve packages)

Let's say you add the pirate package with `poe add --repo=pingpong-templates/hub simple-pirate`.

First this downloads the simple-pirate package to pirate

Then this adds a `poetry` path dependency, which gets picked up from `add_package_routes`.