.PHONY: format lint tests integration_tests

format:
	black .
	isort .

lint:
	mypy .
	black . --check
	isort . --check
	flake8 .

tests:
	pytest tests/unit_tests

integration_tests:
	pytest tests/integration_tests
