# Contributing to LangChain

Hi there! Thank you for even being interested in contributing to LangChain.
As an open source project in a rapidly developing field, we are extremely open
to contributions, whether it be in the form of a new feature, improved infra, or better documentation.

To contribute to this project, please follow a ["fork and pull request"](https://docs.github.com/en/get-started/quickstart/contributing-to-projects) workflow.
Please do not try to push directly to this repo unless you are maintainer.

## 🗺️Contributing Guidelines

### 🚩GitHub Issues

Our [issues](https://github.com/hwchase17/langchain/issues) page is kept up to date
with bugs, improvements, and feature requests. There is a taxonomy of labels to help
with sorting and discovery of issues of interest. These include:

- prompts: related to prompt tooling/infra.
- llms: related to LLM wrappers/tooling/infra.
- chains
- utilities: related to different types of utilities to integrate with (Python, SQL, etc.).
- agents
- memory
- applications: related to example applications to build

If you start working on an issue, please assign it to yourself.

If you are adding an issue, please try to keep it focused on a single modular bug/improvement/feature.
If the two issues are related, or blocking, please link them rather than keep them as one single one.

We will try to keep these issues as up to date as possible, though
with the rapid rate of develop in this field some may get out of date.
If you notice this happening, please just let us know.

### 🙋Getting Help

Although we try to have a developer setup to make it as easy as possible for others to contribute (see below)
it is possible that some pain point may arise around environment setup, linting, documentation, or other.
Should that occur, please contact a maintainer! Not only do we want to help get you unblocked,
but we also want to make sure that the process is smooth for future contributors.

In a similar vein, we do enforce certain linting, formatting, and documentation standards in the codebase.
If you are finding these difficult (or even just annoying) to work with,
feel free to contact a maintainer for help - we do not want these to get in the way of getting
good code into the codebase.

### 🏭Release process

As of now, LangChain has an ad hoc release process: releases are cut with high frequency via by
a developer and published to [PyPI](https://pypi.org/project/langchain/).

LangChain follows the [semver](https://semver.org/) versioning standard. However, as pre-1.0 software,
even patch releases may contain [non-backwards-compatible changes](https://semver.org/#spec-item-4).

If your contribution has made its way into a release, we will want to give you credit on Twitter (only if you want though)!
If you have a Twitter account you would like us to mention, please let us know in the PR or in another manner.

## 🚀Quick Start

This project uses [Poetry](https://python-poetry.org/) as a dependency manager. Check out Poetry's [documentation on how to install it](https://python-poetry.org/docs/#installation) on your system before proceeding.

❗Note: If you use `Conda` or `Pyenv` as your environment / package manager, avoid dependency conflicts by doing the following first:
1. *Before installing Poetry*, create and activate a new Conda env (e.g. `conda create -n langchain python=3.9`)
2. Install Poetry (see above)
3. Tell Poetry to use the virtualenv python environment (`poetry config virtualenvs.prefer-active-python true`)
4. Continue with the following steps.

To install requirements:

```bash
poetry install -E all
```

This will install all requirements for running the package, examples, linting, formatting, tests, and coverage. Note the `-E all` flag will install all optional dependencies necessary for integration testing.

Now, you should be able to run the common tasks in the following section.

## ✅Common Tasks

### Code Formatting

Formatting for this project is done via a combination of [Black](https://black.readthedocs.io/en/stable/) and [isort](https://pycqa.github.io/isort/).

To run formatting for this project:

```bash
make format
```

### Linting

Linting for this project is done via a combination of [Black](https://black.readthedocs.io/en/stable/), [isort](https://pycqa.github.io/isort/), [flake8](https://flake8.pycqa.org/en/latest/), and [mypy](http://mypy-lang.org/).

To run linting for this project:

```bash
make lint
```

We recognize linting can be annoying - if you do not want to do it, please contact a project maintainer, and they can help you with it. We do not want this to be a blocker for good code getting contributed.

### Coverage

Code coverage (i.e. the amount of code that is covered by unit tests) helps identify areas of the code that are potentially more or less brittle.

To get a report of current coverage, run the following:

```bash
make coverage
```

### Testing

Unit tests cover modular logic that does not require calls to outside APIs.

To run unit tests:

```bash
make tests
```

If you add new logic, please add a unit test.

Integration tests cover logic that requires making calls to outside APIs (often integration with other services).

To run integration tests:

```bash
make integration_tests
```

If you add support for a new external API, please add a new integration test.

### Adding a Jupyter Notebook

If you are adding a Jupyter notebook example, you'll want to install the optional `dev` dependencies.

To install dev dependencies:

```bash
poetry install --with dev
```

Launch a notebook:

```bash
poetry run jupyter notebook
```

When you run `poetry install`, the `langchain` package is installed as editable in the virtualenv, so your new logic can be imported into the notebook.

## Documentation

### Contribute Documentation

Docs are largely autogenerated by [sphinx](https://www.sphinx-doc.org/en/master/) from the code.

For that reason, we ask that you add good documentation to all classes and methods.

Similar to linting, we recognize documentation can be annoying. If you do not want to do it, please contact a project maintainer, and they can help you with it. We do not want this to be a blocker for good code getting contributed.

### Build Documentation Locally

Before building the documentation, it is always a good idea to clean the build directory:

```bash
make docs_clean
```

Next, you can run the linkchecker to make sure all links are valid:

```bash
make docs_linkcheck
```

Finally, you can build the documentation as outlined below:

```bash
make docs_build
```
