.PHONY: format lint tests tests_watch integration_tests

coverage:
	poetry run pytest --cov \
		--cov-config=.coveragerc \
		--cov-report xml \
		--cov-report term-missing:skip-covered

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

tests:
	poetry run pytest tests/unit_tests

tests_watch:
	poetry run ptw --now . -- tests/unit_tests

integration_tests:
	poetry run pytest tests/integration_tests
