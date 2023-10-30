# Contributing

Thanks for taking the time to contribute a new template!
We've tried to make this process as simple and painless as possible.
If you need any help at all, please reach out!

To contribute a new template, first fork this repository.
Then clone that fork and pull it down locally.
Set up an appropriate dev environment, and make sure you are in this `templates` directory.

Make sure you have `langchain-cli` installed.

```shell
pip install -U "langchain-cli[serve]"
```

You can then run the following command to create a new skeleton of a package.
By convention, package names should use `-` delimeters (not `_`).

```shell
langchain hub new $PROJECT_NAME
```

You can then edit the contents of the package as you desire.
Note that by default we expect the main chain to be exposed as `chain` in the `__init__.py` file of the package.
You can change this (either the name or the location), but if you do so it is important to update the `tool.langchain`
part of `pyproject.toml`.
For example, if you update the main chain exposed to be called `agent_executor`, then that section should look like:

```text
[tool.langserve]
export_module = "..."
export_attr = "agent_executor"
```

Make sure to add any requirements of the package to `pyproject.toml` (and to remove any that are not used).

Please update the `README.md` file to give some background on your package and how to set it up.

If you want to change the license of your template for whatever, you may! Note that by default it is MIT licensed.

If you want to test out your package at any point in time, you can spin up a LangServe instance directly from the package.
See instructions [here](LAUNCHING_PACKAGE.md) on how to best do that.
