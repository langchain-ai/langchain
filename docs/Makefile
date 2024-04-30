# we build the docs in these stages:
# 1. install quarto and python dependencies
# 2. copy files from "source dir" to "intermediate dir"
# 2. generate files like model feat table, etc in "intermediate dir"
# 3. copy files to their right spots (e.g. langserve readme) in "intermediate dir"
# 4. build the docs from "intermediate dir" to "output dir"

SOURCE_DIR = docs/
INTERMEDIATE_DIR = build/intermediate
OUTPUT_DIR = build/output

PYTHON = .venv/bin/python

clean:
	rm -rf build

vercel-install-deps:
	yum -y update
	yum install gcc bzip2-devel libffi-devel zlib-devel wget tar gzip -y

	# install quarto
	wget -q https://github.com/quarto-dev/quarto-cli/releases/download/v1.3.450/quarto-1.3.450-linux-amd64.tar.gz
	tar -xzf quarto-1.3.450-linux-amd64.tar.gz

install-py-deps:
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install --upgrade uv
	$(PYTHON) -m uv pip install -r vercel_requirements.txt
	$(PYTHON) -m uv pip install -e $(ls ../libs/partners | grep -vE "airbyte|ibm|.md" | xargs -I {} echo "../libs/partners/{}")

generate-files:
	cp -r $(SOURCE_DIR)/* $(INTERMEDIATE_DIR)
	cp ../templates/docs/INDEX.md $(INTERMEDIATE_DIR)/docs/templates/index.md
	cp ../cookbook/README.md $(INTERMEDIATE_DIR)/cookbook.mdx

	$(PYTHON) scripts/model_feat_table.py $(INTERMEDIATE_DIR)

	mkdir -p $(INTERMEDIATE_DIR)/docs/templates
	$(PYTHON) scripts/copy_templates.py $(INTERMEDIATE_DIR)

	wget -q https://raw.githubusercontent.com/langchain-ai/langserve/main/README.md -O $(INTERMEDIATE_DIR)/docs/langserve.md
	$(PYTHON) scripts/resolve_local_links.py $(INTERMEDIATE_DIR)/docs/langserve.md https://github.com/langchain-ai/langserve/tree/main/

	wget -q https://raw.githubusercontent.com/langchain-ai/langgraph/main/README.md -O docs/langgraph.md
	$(PYTHON) scripts/resolve_local_links.py $(INTERMEDIATE_DIR)/docs/langgraph.md https://github.com/langchain-ai/langgraph/tree/main/

	$(PYTHON) scripts/generate_api_reference_links.py --docs_dir docs


build-local: install-py-deps generate-files
	quarto render docs/

build-vercel: vercel-install-deps install-py-deps generate-files
	$(pwd)/quarto-1.3.450/bin/quarto render docs/

build: build-local
