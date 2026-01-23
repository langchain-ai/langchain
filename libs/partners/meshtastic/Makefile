######################
# TESTING
######################

# Run unit tests
test:
	uv run --group test pytest tests/unit_tests

# Run integration tests
test_integration:
	uv run --group test pytest tests/integration_tests

######################
# LINTING AND FORMATTING
######################

# Linting with ruff
lint:
	uv run --group lint ruff check .
	uv run --group lint ruff format . --check

# Auto-fix linting issues
lint_fix:
	uv run --group lint ruff check . --fix
	uv run --group lint ruff format .

# Run all formatting
format:
	uv run --group lint ruff format .
	uv run --group lint ruff check . --fix

######################
# TYPE CHECKING
######################

# Type checking with mypy
check_type:
	uv run --group typing mypy .

######################
# DEVELOPMENT
######################

# Install all dependencies
install:
	uv sync --all-groups

# Clean up
clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache

######################
# SHORTCUTS
######################

all: format lint test
