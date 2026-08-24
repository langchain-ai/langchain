
.PHONY: all lint format test tests integration_tests help

all: help

######################
# LINTING AND FORMATTING
######################

lint format: PYTHON_FILES=.
lint:
	ruff check $(PYTHON_FILES)
	ruff format $(PYTHON_FILES) --diff
	mypy $(PYTHON_FILES)

format:
	ruff format $(PYTHON_FILES)
	ruff check --fix $(PYTHON_FILES)

######################
# TESTING
######################

tests test:
	pytest tests/unit_tests

integration_tests:
	pytest tests/integration_tests

######################
# HELP
######################

help:
	@echo '----'
	@echo 'lint                         - run linters'
	@echo 'format                       - run formatters'
	@echo 'tests                        - run unit tests'
	@echo 'integration_tests            - run integration tests (requires DECODO_API_TOKEN)'
