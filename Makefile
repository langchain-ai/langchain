.PHONY: format lint tests

format:
	black .
	isort .

lint: 
	black . --check
	isort . --check
	flake8 .
	mypy .

tests:
	pytest tests