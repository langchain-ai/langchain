# __package_name_last__

TODO: What does this package do

## Environment Setup

TODO: What environment variables need to be set (if any)

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U "langchain-cli[serve]"
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain serve new my-app
cd my-app
langchain serve add __package_name_last__
```

If you want to add this to an existing project, you can just run:

```shell
langchain serve add __package_name_last__
```

If you are inside this directory, then you can spin up a LangServe instance directly by:

```shell
langchain hub start
```
