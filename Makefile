# Makefile for development convenience
.PHONY: install run lint test format

install:
	@echo "Install dependencies (use uv or pip)"
	@echo "Recommended: uv sync --all-groups"

run:
	@echo "Run Streamlit app"
	streamlit run src/content_growth_agent/app/streamlit_app.py

lint:
	ruff . || true

test:
	pytest -q

format:
	ruff format .
