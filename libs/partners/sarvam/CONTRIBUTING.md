# Contributing to langchain-sarvam

Thank you for your interest in contributing to langchain-sarvam! This document provides guidelines for contributing to this package.

## Development Setup

1. Clone the repository and navigate to the package directory:
```bash
cd libs/partners/sarvam
```

2. Install dependencies using Poetry:
```bash
poetry install --with test,lint,typing,dev
```

3. Set up your Sarvam API key for testing:
```bash
export SARVAM_API_KEY="your-api-key"
```

## Running Tests

### Unit Tests
```bash
make test
# or
poetry run pytest tests/unit_tests
```

### Integration Tests
Integration tests require a valid Sarvam API key:
```bash
make integration_tests
# or
poetry run pytest tests/integration_tests
```

### All Tests
```bash
poetry run pytest tests/
```

## Code Quality

### Linting
```bash
make lint
```

This will run:
- `ruff` for code style checking
- `mypy` for type checking

### Formatting
```bash
make format
```

This will automatically format your code using `ruff`.

### Spell Checking
```bash
make spell_check
# To automatically fix spelling issues:
make spell_fix
```

## Before Submitting a PR

1. **Run all tests**: Ensure all unit and integration tests pass
```bash
poetry run pytest tests/
```

2. **Run linting**: Fix any linting errors
```bash
make lint
make format
```

3. **Check imports**: Verify imports are correct
```bash
make check_imports
```

4. **Update documentation**: If you've added new features, update:
   - README.md
   - Docstrings in the code
   - Example notebooks if applicable

## Code Style Guidelines

- Follow PEP 8 style guidelines
- Use type hints for all function parameters and return values
- Write descriptive docstrings for all public methods and classes
- Keep functions focused and single-purpose
- Add comments for complex logic

## Testing Guidelines

- Write unit tests for all new functionality
- Ensure tests are isolated and don't depend on external services (use mocking)
- Integration tests should be marked with `@pytest.mark.scheduled`
- Test edge cases and error conditions

## Pull Request Process

1. Fork the repository and create a new branch for your feature
2. Make your changes following the guidelines above
3. Ensure all tests pass and code is properly formatted
4. Update documentation as needed
5. Submit a pull request with a clear description of changes

## Common Issues

### CI/CD Failures

If you encounter CI failures:

1. **Linting errors** (`lint` job failing):
   - Run `make lint` locally
   - Fix any `ruff` or `mypy` errors
   - Run `make format` to auto-format code

2. **Test failures**:
   - Run tests locally: `poetry run pytest`
   - Check if API key is properly set for integration tests
   - Review test output for specific failures

3. **Import errors**:
   - Verify all imports are from allowed packages
   - Run `make check_imports`

## Questions or Problems?

If you have questions or run into issues:
- Check existing GitHub issues
- Create a new issue with a clear description
- Join the LangChain Discord community

Thank you for contributing!
