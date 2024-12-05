lint lint_diff:
	poetry run poe lint

test:
	poetry run poe test

PYTHON = .venv/bin/python

# custom integration testing for cli integration flow
integration_test integration_tests:
	rm -rf .integration_test
	mkdir .integration_test
	cd .integration_test
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade poetry
	$(PYTHON) -m pip install -e ../
	$(PYTHON) -m langchain integration new --name parrot-link --name-class ParrotLink
	cd langchain-parrot-link


format:
	poetry run poe format
