# Contributing to LangChain

Hi there! Thank you for even being interested in contributing to LangChain.
As an open source project in a rapidly developing field, we are extremely open
to contributions, whether they be in the form of new features, improved infra, better documentation, or bug fixes.

## 🗺️ Guidelines

### 👩‍💻 Contributing Code

To contribute to this project, please follow a ["fork and pull request"](https://docs.github.com/en/get-started/quickstart/contributing-to-projects) workflow.
Please do not try to push directly to this repo unless you are maintainer.

Please follow the checked-in pull request template when opening pull requests. Note related issues and tag relevant
maintainers.

Pull requests cannot land without passing the formatting, linting and testing checks first. See
[Common Tasks](#-common-tasks) for how to run these checks locally.

It's essential that we maintain great documentation and testing. If you:
- Fix a bug
  - Add a relevant unit or integration test when possible. These live in `tests/unit_tests` and `tests/integration_tests`.
- Make an improvement
  - Update any affected example notebooks and documentation. These lives in `docs`.
  - Update unit and integration tests when relevant.
- Add a feature
  - Add a demo notebook in `docs/modules`.
  - Add unit and integration tests.

We're a small, building-oriented team. If there's something you'd like to add or change, opening a pull request is the
best way to get our attention.

### 🚩GitHub Issues

Our [issues](https://github.com/hwchase17/langchain/issues) page is kept up to date
with bugs, improvements, and feature requests. 

There is a taxonomy of labels to help with sorting and discovery of issues of interest. Please use these to help
organize issues.

If you start working on an issue, please assign it to yourself.

If you are adding an issue, please try to keep it focused on a single, modular bug/improvement/feature.
If two issues are related, or blocking, please link them rather than combining them.

We will try to keep these issues as up to date as possible, though
with the rapid rate of develop in this field some may get out of date.
If you notice this happening, please let us know.

### 🙋Getting Help

Our goal is to have the simplest developer setup possible. Should you experience any difficulty getting setup, please
contact a maintainer! Not only do we want to help get you unblocked, but we also want to make sure that the process is
smooth for future contributors.

In a similar vein, we do enforce certain linting, formatting, and documentation standards in the codebase.
If you are finding these difficult (or even just annoying) to work with, feel free to contact a maintainer for help -
we do not want these to get in the way of getting good code into the codebase.

## 🚀 Quick Start

> **Note:** You can run this repository locally (which is described below) or in a [development container](https://containers.dev/) (which is described in the [.devcontainer folder](https://github.com/hwchase17/langchain/tree/master/.devcontainer)).

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

❗Note: If you're running Poetry 1.4.1 and receive a `WheelFileValidationError` for `debugpy` during installation, you can try either downgrading to Poetry 1.4.0 or disabling "modern installation" (`poetry config installer.modern-installation false`) and re-install requirements. See [this `debugpy` issue](https://github.com/microsoft/debugpy/issues/1246) for more details.

Now, you should be able to run the common tasks in the following section. To double check, run `make test`, all tests should pass. If they don't you may need to pip install additional dependencies, such as `numexpr` and `openapi_schema_pydantic`.

## ✅ Common Tasks

Type `make` for a list of common tasks.

### Code Formatting

Formatting for this project is done via a combination of [Black](https://black.readthedocs.io/en/stable/) and [isort](https://pycqa.github.io/isort/).

To run formatting for this project:

```bash
make format
```

Additionally, you can run the formatter only on the files that have been modified in your current branch as compared to the master branch using the format_diff command:

```bash
make format_diff
```

This is especially useful when you have made changes to a subset of the project and want to ensure your changes are properly formatted without affecting the rest of the codebase.

### Linting

Linting for this project is done via a combination of [Black](https://black.readthedocs.io/en/stable/), [isort](https://pycqa.github.io/isort/), [flake8](https://flake8.pycqa.org/en/latest/), and [mypy](http://mypy-lang.org/).

To run linting for this project:

```bash
make lint
```

In addition, you can run the linter only on the files that have been modified in your current branch as compared to the master branch using the lint_diff command:

```bash
make lint_diff
```

This can be very helpful when you've made changes to only certain parts of the project and want to ensure your changes meet the linting standards without having to check the entire codebase.

We recognize linting can be annoying - if you do not want to do it, please contact a project maintainer, and they can help you with it. We do not want this to be a blocker for good code getting contributed.

### Spellcheck

Spellchecking for this project is done via [codespell](https://github.com/codespell-project/codespell).
Note that `codespell` finds common typos, so could have false-positive (correctly spelled but rarely used) and false-negatives (not finding misspelled) words.

To check spelling for this project:

```bash
make spell_check
```

To fix spelling in place:

```bash
make spell_fix
```

If codespell is incorrectly flagging a word, you can skip spellcheck for that word by adding it to the codespell config in the `pyproject.toml` file.

```python
[tool.codespell]
...
# Add here:
ignore-words-list = 'momento,collison,ned,foor,reworkd,parth,whats,aapply,mysogyny,unsecure'
```

### Coverage

Code coverage (i.e. the amount of code that is covered by unit tests) helps identify areas of the code that are potentially more or less brittle.

To get a report of current coverage, run the following:

```bash
make coverage
```

### Working with Optional Dependencies

Langchain relies heavily on optional dependencies to keep the Langchain package lightweight.

If you're adding a new dependency to Langchain, assume that it will be an optional dependency, and
that most users won't have it installed.

Users that do not have the dependency installed should be able to **import** your code without
any side effects (no warnings, no errors, no exceptions). 

To introduce the dependency to the pyproject.toml file correctly, please do the following: 

1. Add the dependency to the main group as an optional dependency
  ```bash
  poetry add --optional [package_name]
  ```
2. Open pyproject.toml and add the dependency to the `extended_testing` extra
3. Relock the poetry file to update the extra.
  ```bash
  poetry lock --no-update
  ```
4. Add a unit test that the very least attempts to import the new code. Ideally the unit
test makes use of lightweight fixtures to test the logic of the code.
5. Please use the `@pytest.mark.requires(package_name)` decorator for any tests that require the dependency.

### Testing

See section about optional dependencies.

#### Unit Tests

Unit tests cover modular logic that does not require calls to outside APIs.

To run unit tests:

```bash
make test
```

To run unit tests in Docker:

```bash
make docker_tests
```

If you add new logic, please add a unit test.



#### Integration Tests

Integration tests cover logic that requires making calls to outside APIs (often integration with other services).

**warning** Almost no tests should be integration tests. 

  Tests that require making network connections make it difficult for other
  developers to test the code.

  Instead favor relying on `responses` library and/or mock.patch to mock
  requests using small fixtures.

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

The docs directory contains Documentation and API Reference.

Documentation is built using [Docusaurus 2](https://docusaurus.io/).

API Reference are largely autogenerated by [sphinx](https://www.sphinx-doc.org/en/master/) from the code.
For that reason, we ask that you add good documentation to all classes and methods.

Similar to linting, we recognize documentation can be annoying. If you do not want to do it, please contact a project maintainer, and they can help you with it. We do not want this to be a blocker for good code getting contributed.

### Build Documentation Locally

In the following commands, the prefix `api_` indicates that those are operations for the API Reference.

Before building the documentation, it is always a good idea to clean the build directory:

```bash
make docs_clean
make api_docs_clean
```

Next, you can build the documentation as outlined below:

```bash
make docs_build
make api_docs_build
```

Finally, you can run the linkchecker to make sure all links are valid:

```bash
make docs_linkcheck
make api_docs_linkcheck
```

## 🏭 Release Process

As of now, LangChain has an ad hoc release process: releases are cut with high frequency by
a developer and published to [PyPI](https://pypi.org/project/langchain/).

LangChain follows the [semver](https://semver.org/) versioning standard. However, as pre-1.0 software,
even patch releases may contain [non-backwards-compatible changes](https://semver.org/#spec-item-4).

### 🌟 Recognition

If your contribution has made its way into a release, we will want to give you credit on Twitter (only if you want though)!
If you have a Twitter account you would like us to mention, please let us know in the PR or in another manner.

