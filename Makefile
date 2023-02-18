.PHONY: all clean format lint test tests test_watch integration_tests help

all: help
	
coverage:
	poetry run pytest --cov \
		--cov-config=.coveragerc \
		--cov-report xml \
		--cov-report term-missing:skip-covered

clean: docs_clean

docs_build:
	cd docs && poetry run make html

docs_clean:
	cd docs && poetry run make clean

docs_linkcheck:
	poetry run linkchecker docs/_build/html/index.html

format:
	poetry run black .
	poetry run isort .

lint:
	poetry run mypy .
	poetry run black . --check
	poetry run isort . --check
	poetry run flake8 .

test:
	poetry run pytest tests/unit_tests

tests:
	poetry run pytest tests/unit_tests

test_watch:
	poetry run ptw --now . -- tests/unit_tests

integration_tests:
	poetry run pytest tests/integration_tests

help:
	@echo '----'
	@echo 'coverage            - run unit tests and generate coverage report'
	@echo 'docs_build          - build the documentation'
	@echo 'docs_clean          - clean the documentation build artifacts'
	@echo 'docs_linkcheck      - run linkchecker on the documentation'
	@echo 'format              - run code formatters'
	@echo 'lint                - run linters'
	@echo 'test                - run unit tests'
	@echo 'test_watch          - run unit tests in watch mode'
	@echo 'integration_tests   - run integration tests'
