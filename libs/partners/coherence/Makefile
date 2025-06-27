.PHONY: install lint format test clean

install:
	uv pip install -e .[lint,typing,test]

lint:
	ruff check langchain_coherence tests

format:
	ruff format langchain_coherence tests

fix: format lint

test:
	pytest

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .uv __pycache__ *.egg-info build dist

