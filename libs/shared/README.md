# Shared Tool Configurations

This directory contains shared configuration files for development tools used across LangChain packages.

## Purpose

To centralize common tool configurations (mypy, ruff) that are duplicated across multiple `pyproject.toml` files in the monorepo. This improves maintainability and consistency.

## Files

- `mypy.ini` - Shared mypy type checking configuration
- `ruff.toml` - Shared ruff linting and formatting configuration

## Usage

### Mypy

Packages should reference the shared mypy config in their `pyproject.toml`:

```toml
[tool.mypy]
# See ../shared/mypy.ini for base config (or ../../shared/mypy.ini for partners)
plugins = ["pydantic.mypy"]
strict = true
enable_error_code = "deprecated"

# Package-specific overrides can be added here
```

**Note:** Mypy doesn't support native config file extension in `pyproject.toml`, so packages maintain their own `[tool.mypy]` section that should match the shared config. Package-specific overrides can be added as needed.

### Ruff

Packages should reference the shared ruff config in their `pyproject.toml`:

```toml
[tool.ruff.format]
# See ../shared/ruff.toml for base config (or ../../shared/ruff.toml for partners)
docstring-code-format = true

[tool.ruff.lint]
# ... config should match shared/ruff.toml with package-specific overrides
```

**Note:** Ruff doesn't support native config file extension in `pyproject.toml`, so packages maintain their own `[tool.ruff]` section that should match the shared config. Package-specific overrides (like per-file ignores) can be added as needed.

## Updating Configurations

When updating tool configurations:

1. Update the shared config file (`mypy.ini` or `ruff.toml`)
2. Update all package `pyproject.toml` files to match the shared config
3. Keep package-specific overrides where necessary

## Path References

- From `libs/core/` or `libs/langchain_v1/`: `../shared/`
- From `libs/partners/*/`: `../../shared/`
- From `libs/text-splitters/` or `libs/standard-tests/`: `../shared/`

## Future Improvements

- Consider creating a script to sync shared configs to package configs
- Explore tooling to validate that package configs match shared configs
- Investigate if future versions of mypy/ruff support native config extension

