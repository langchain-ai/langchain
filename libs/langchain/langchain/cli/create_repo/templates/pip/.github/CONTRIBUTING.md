# Contributing to ____project_name

Hi there! Thank you for even being interested in contributing to ____project_name.

## 🚀 Quick Start

To install requirements:

```bash
poetry install -e ".[dev]"
```

This will install all requirements for running the package, examples, linting, formatting, tests, and coverage.

Now, you should be able to run the common tasks in the following section. To double check, run `make test`, all tests should pass.

## ✅ Common Tasks

Type `make` for a list of common tasks.

### Code Formatting

Formatting for this project is done via a combination of [Black](https://black.readthedocs.io/en/stable/) and [isort](https://pycqa.github.io/isort/).

To run formatting for this project:

```bash
make format
```

Additionally, you can run the formatter only on the files that have been modified in your current branch as compared to the main branch using the format_diff command:

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

In addition, you can run the linter only on the files that have been modified in your current branch as compared to the main branch using the lint_diff command:

```bash
make lint_diff
```

This can be very helpful when you've made changes to only certain parts of the project and want to ensure your changes meet the linting standards without having to check the entire codebase.

We recognize linting can be annoying - if you do not want to do it, please contact a project maintainer, and they can help you with it. We do not want this to be a blocker for good code getting contributed.

### Testing

To run unit tests:

```bash
make test
```

If you add new logic, please add a unit test.

## 🏭 Release Process

____project_name follows the [semver](https://semver.org/) versioning standard.

To use the [automated release workflow](./workflows/release.yml) you'll need to set up a PyPI account and [create an API token](https://pypi.org/help/#apitoken). Configure the API token for this GitHub repo by going to settings -> security -> secrets -> actions, creating the `PYPI_API_TOKEN` variable and setting the value to be your PyPI API token.

Once that's set up, you can release a new version of the package by opening a PR that:
1. updates the package version in the [pyproject.toml file](../pyproject.toml),
2. labels the PR with a `release` tag.
When the PR is merged into main, a new release will be created.

