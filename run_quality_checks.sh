poetry run ruff format libs/partners/google-vertexai/
poetry run codespell --toml pyproject.toml -w libs/partners/google-vertexai/ 
poetry run mypy libs/partners/google-vertexai/langchain_google_vertexai --cache-dir .mypy_cache