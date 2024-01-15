lint lint_diff lint_package lint_tests:
	poetry run poe lint

test:
	poetry run poe test

format:
	poetry run poe format

lock:
	@echo "Locking dependencies..."
	@for d in $$(find . -name pyproject.toml -exec dirname {} \;); do \
		(cd $$d && poetry lock); \
	done

check:
	@echo "Checking dependencies..."
	@for d in $$(find . -name pyproject.toml -exec dirname {} \;); do \
		(cd $$d && echo "Checking $$d" && poetry check); \
	done