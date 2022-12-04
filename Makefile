.PHONY: format lint tests integration_tests

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

integration_tests:
	poetry run pytest tests/integration_tests
