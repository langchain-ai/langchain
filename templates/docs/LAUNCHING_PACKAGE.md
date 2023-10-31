# Launching LangServe from a Package

You can also launch LangServe directly from a package, without having to pull it into a project.
This can be useful when you are developing a package and want to test it quickly.
The downside of this is that it gives you a little less control over how the LangServe APIs are configured,
which is why for proper projects we recommend creating a full project.

In order to do this, first change your working directory to the package itself.
For example, if you are currently in this `templates` module, you can go into the `pirate-speak` package with:

```shell
cd pirate-speak
```

Inside this package there is a `pyproject.toml` file.
This file contains a `tool.langchain` section that contains information on how this package should be used.
For example, in `pirate-speak` we see:

```text
[tool.langserve]
export_module = "pirate_speak.chain"
export_attr = "chain"
```

This information can be used to launch a LangServe instance automatically.
In order to do this, first make sure the CLI is installed:

```shell
pip install -U "langchain-cli[serve]"
```

You can then run:

```shell
langchain template serve
```

This will spin up endpoints, documentation, and playground for this chain.
For example, you can access the playground at [http://127.0.0.1:8000/playground/](http://127.0.0.1:8000/playground/)

![playground.png](playground.png)
