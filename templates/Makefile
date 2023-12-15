lint lint_diff lint_package lint_tests:
	poetry run poe lint

test:
	poetry run poe test

format:
	poetry run poe format

lock:
	for d in $$(find . -name pyproject.toml -exec dirname {} \;); do \
		(cd $$d && poetry lock --no-update); \
	done