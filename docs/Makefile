# we build the docs in these stages:
# 1. install vercel and python dependencies
# 2. copy files from "source dir" to "intermediate dir"
# 2. generate files like model feat table, etc in "intermediate dir"
# 3. copy files to their right spots (e.g. langserve readme) in "intermediate dir"
# 4. build the docs from "intermediate dir" to "output dir"

SOURCE_DIR = docs/
INTERMEDIATE_DIR = build/intermediate/docs

OUTPUT_NEW_DIR = build/output-new
OUTPUT_NEW_DOCS_DIR = $(OUTPUT_NEW_DIR)/docs

PYTHON = .venv/bin/python

PARTNER_DEPS_LIST := $(shell find ../libs/partners -mindepth 1 -maxdepth 1 -type d -exec test -e "{}/pyproject.toml" \; -print | grep -vE "airbyte|ibm|couchbase" | tr '\n' ' ')

PORT ?= 3001

clean:
	rm -rf build

install-vercel-deps:
	yum -y update
	yum install gcc bzip2-devel libffi-devel zlib-devel wget tar gzip rsync -y

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

	$(PYTHON) scripts/model_feat_table.py $(INTERMEDIATE_DIR)

	$(PYTHON) scripts/document_loader_feat_table.py $(INTERMEDIATE_DIR)

	$(PYTHON) scripts/copy_templates.py $(INTERMEDIATE_DIR)

	wget -q https://raw.githubusercontent.com/langchain-ai/langserve/main/README.md -O $(INTERMEDIATE_DIR)/langserve.md
	$(PYTHON) scripts/resolve_local_links.py $(INTERMEDIATE_DIR)/langserve.md https://github.com/langchain-ai/langserve/tree/main/

copy-infra:
	mkdir -p $(OUTPUT_NEW_DIR)
	cp -r src $(OUTPUT_NEW_DIR)
	cp vercel.json $(OUTPUT_NEW_DIR)
	cp babel.config.js $(OUTPUT_NEW_DIR)
	cp -r data $(OUTPUT_NEW_DIR)
	cp docusaurus.config.js $(OUTPUT_NEW_DIR)
	cp package.json $(OUTPUT_NEW_DIR)
	cp sidebars.js $(OUTPUT_NEW_DIR)
	cp -r static $(OUTPUT_NEW_DIR)
	cp yarn.lock $(OUTPUT_NEW_DIR)

render:
	$(PYTHON) scripts/notebook_convert.py $(INTERMEDIATE_DIR) $(OUTPUT_NEW_DOCS_DIR)

md-sync:
	rsync -avm --include="*/" --include="*.mdx" --include="*.md" --include="*.png" --include="*/_category_.yml" --exclude="*" $(INTERMEDIATE_DIR)/ $(OUTPUT_NEW_DOCS_DIR)

generate-references:
	$(PYTHON) scripts/generate_api_reference_links.py --docs_dir $(OUTPUT_NEW_DOCS_DIR)

build: install-py-deps generate-files copy-infra render md-sync

vercel-build: install-vercel-deps build generate-references
	rm -rf docs
	mv $(OUTPUT_NEW_DOCS_DIR) docs
	rm -rf build
	yarn run docusaurus build
	mv build v0.2
	mkdir build
	mv v0.2 build
	mv build/v0.2/404.html build

start:
	cd $(OUTPUT_NEW_DIR) && yarn && yarn start --port=$(PORT)
