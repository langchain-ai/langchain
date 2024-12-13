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

PORT ?= 3001

clean:
	rm -rf build

install-vercel-deps:
	yum -y -q update
	yum -y -q install gcc bzip2-devel libffi-devel zlib-devel wget tar gzip rsync -y

install-py-deps:
	python3 -m venv .venv
	$(PYTHON) -m pip install -q --upgrade pip
	$(PYTHON) -m pip install -q --upgrade uv
	$(PYTHON) -m uv pip install -q --pre -r vercel_requirements.txt
	$(PYTHON) -m uv pip install -q --pre $$($(PYTHON) scripts/partner_deps_list.py)

generate-files:
	mkdir -p $(INTERMEDIATE_DIR)
	cp -rp $(SOURCE_DIR)/* $(INTERMEDIATE_DIR)

	$(PYTHON) scripts/tool_feat_table.py $(INTERMEDIATE_DIR)

	$(PYTHON) scripts/kv_store_feat_table.py $(INTERMEDIATE_DIR)

	$(PYTHON) scripts/partner_pkg_table.py $(INTERMEDIATE_DIR)

	curl https://raw.githubusercontent.com/langchain-ai/langserve/main/README.md | sed 's/<=/\&lt;=/g' > $(INTERMEDIATE_DIR)/langserve.md
	cp ../SECURITY.md $(INTERMEDIATE_DIR)/security.md
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
	cp -r ../libs/cli/langchain_cli/integration_template $(OUTPUT_NEW_DIR)/src/theme
	cp yarn.lock $(OUTPUT_NEW_DIR)

render:
	$(PYTHON) scripts/notebook_convert.py $(INTERMEDIATE_DIR) $(OUTPUT_NEW_DOCS_DIR)

md-sync:
	rsync -avmq --include="*/" --include="*.mdx" --include="*.md" --include="*.png" --include="*/_category_.yml" --exclude="*" $(INTERMEDIATE_DIR)/ $(OUTPUT_NEW_DOCS_DIR)

append-related:
	$(PYTHON) scripts/append_related_links.py $(OUTPUT_NEW_DOCS_DIR)

generate-references:
	$(PYTHON) scripts/generate_api_reference_links.py --docs_dir $(OUTPUT_NEW_DOCS_DIR)

update-md: generate-files md-sync

build: install-py-deps generate-files copy-infra render md-sync append-related

vercel-build: install-vercel-deps build generate-references
	rm -rf docs
	mv $(OUTPUT_NEW_DOCS_DIR) docs
	cp -r ../libs/cli/langchain_cli/integration_template src/theme
	rm -rf build
	mkdir static/api_reference
	git clone --depth=1 https://github.com/langchain-ai/langchain-api-docs-html.git
	mv langchain-api-docs-html/api_reference_build/html/* static/api_reference/
	rm -rf langchain-api-docs-html
	NODE_OPTIONS="--max-old-space-size=5000" yarn run docusaurus build

start:
	cd $(OUTPUT_NEW_DIR) && yarn && yarn start --port=$(PORT)
