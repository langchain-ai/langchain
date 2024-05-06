# we build the docs in these stages:
# 1. install quarto and python dependencies
# 2. copy files from "source dir" to "intermediate dir"
# 2. generate files like model feat table, etc in "intermediate dir"
# 3. copy files to their right spots (e.g. langserve readme) in "intermediate dir"
# 4. build the docs from "intermediate dir" to "output dir"

SOURCE_DIR = docs/
INTERMEDIATE_DIR = build/intermediate/docs
OUTPUT_DIR = build/output
OUTPUT_DOCS_DIR = $(OUTPUT_DIR)/docs

PYTHON = .venv/bin/python

QUARTO_CMD ?= quarto

PARTNER_DEPS_LIST := $(shell ls -1 ../libs/partners | grep -vE "airbyte|ibm" | xargs -I {} echo "../libs/partners/{}" | tr '\n' ' ')

PORT ?= 3001

clean:
	rm -rf build

install-vercel-deps:
	yum -y update
	yum install gcc bzip2-devel libffi-devel zlib-devel wget tar gzip rsync -y

	wget -q https://github.com/quarto-dev/quarto-cli/releases/download/v1.3.450/quarto-1.3.450-linux-amd64.tar.gz
	tar -xzf quarto-1.3.450-linux-amd64.tar.gz

install-py-deps:
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install --upgrade uv
	$(PYTHON) -m uv pip install -r vercel_requirements.txt
	$(PYTHON) -m uv pip install --editable $(PARTNER_DEPS_LIST)

generate-files:
	mkdir -p $(INTERMEDIATE_DIR)
	cp -r $(SOURCE_DIR)/* $(INTERMEDIATE_DIR)
	mkdir -p $(INTERMEDIATE_DIR)/templates
	cp ../templates/docs/INDEX.md $(INTERMEDIATE_DIR)/templates/index.md
	cp ../cookbook/README.md $(INTERMEDIATE_DIR)/cookbook.mdx

	$(PYTHON) scripts/model_feat_table.py $(INTERMEDIATE_DIR)

	$(PYTHON) scripts/copy_templates.py $(INTERMEDIATE_DIR)

	wget -q https://raw.githubusercontent.com/langchain-ai/langserve/main/README.md -O $(INTERMEDIATE_DIR)/langserve.md
	$(PYTHON) scripts/resolve_local_links.py $(INTERMEDIATE_DIR)/langserve.md https://github.com/langchain-ai/langserve/tree/main/

	wget -q https://raw.githubusercontent.com/langchain-ai/langgraph/main/README.md -O $(INTERMEDIATE_DIR)/langgraph.md
	$(PYTHON) scripts/resolve_local_links.py $(INTERMEDIATE_DIR)/langgraph.md https://github.com/langchain-ai/langgraph/tree/main/

	$(PYTHON) scripts/generate_api_reference_links.py --docs_dir $(INTERMEDIATE_DIR)

copy-infra:
	mkdir -p $(OUTPUT_DIR)
	cp -r src $(OUTPUT_DIR)
	cp vercel.json $(OUTPUT_DIR)
	cp babel.config.js $(OUTPUT_DIR)
	cp -r data $(OUTPUT_DIR)
	cp docusaurus.config.js $(OUTPUT_DIR)
	cp package.json $(OUTPUT_DIR)
	cp sidebars.js $(OUTPUT_DIR)
	cp -r static $(OUTPUT_DIR)
	cp yarn.lock $(OUTPUT_DIR)

quarto-render:
	$(QUARTO_CMD) render $(INTERMEDIATE_DIR) --output-dir $(OUTPUT_DOCS_DIR) --no-execute
	mv $(OUTPUT_DOCS_DIR)/$(INTERMEDIATE_DIR)/* $(OUTPUT_DOCS_DIR)
	rm -rf $(OUTPUT_DOCS_DIR)/build

md-sync:
	rsync -avm --include="*/" --include="*.mdx" --include="*.md" --exclude="*" $(INTERMEDIATE_DIR)/ $(OUTPUT_DOCS_DIR)

build: install-py-deps generate-files copy-infra quarto-render md-sync

start:
	cd $(OUTPUT_DIR) && yarn && yarn start --port=$(PORT)
