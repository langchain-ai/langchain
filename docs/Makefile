# We build the docs in these stages:
# 1. Install vercel and python dependencies
# 2. Copy files from "source dir" to "intermediate dir"
# 2. Generate files like model feat table, etc in "intermediate dir"
# 3. Copy files to their right spots (e.g. langserve readme) in "intermediate dir"
# 4. Build the docs from "intermediate dir" to "output dir"

SOURCE_DIR = docs/
INTERMEDIATE_DIR = build/intermediate/docs

OUTPUT_NEW_DIR = build/output-new
OUTPUT_NEW_DOCS_DIR = $(OUTPUT_NEW_DIR)/docs

PYTHON = .venv/bin/python

PORT ?= 3001

clean:
	rm -rf build

clean-cache:
	rm -rf build .venv/deps_installed

install-vercel-deps:
	yum -y -q update
	yum -y -q install gcc bzip2-devel libffi-devel zlib-devel wget tar gzip rsync -y

install-py-deps:
	@echo "📦 Installing Python dependencies..."
	@if [ ! -d .venv ]; then python3 -m venv .venv; fi
	@if [ ! -f .venv/deps_installed ]; then \
		$(PYTHON) -m pip install -q --upgrade pip --disable-pip-version-check; \
		$(PYTHON) -m pip install -q --upgrade uv; \
		$(PYTHON) -m uv pip install -q --pre -r vercel_requirements.txt; \
		$(PYTHON) -m uv pip install -q --pre $$($(PYTHON) scripts/partner_deps_list.py) --overrides vercel_overrides.txt; \
		touch .venv/deps_installed; \
	fi
	@echo "✅ Dependencies installed"

generate-files:
	@echo "📄 Generating documentation files..."
	mkdir -p $(INTERMEDIATE_DIR)
	cp -rp $(SOURCE_DIR)/* $(INTERMEDIATE_DIR)
	@if [ ! -f build/langserve_readme_cache.md ] || [ $$(find build/langserve_readme_cache.md -mtime +1 -print) ]; then \
		echo "🌐 Downloading LangServe README..."; \
		curl -s https://raw.githubusercontent.com/langchain-ai/langserve/main/README.md | sed 's/<=/\&lt;=/g' > build/langserve_readme_cache.md; \
	fi
	cp build/langserve_readme_cache.md $(INTERMEDIATE_DIR)/langserve.md
	cp ../SECURITY.md $(INTERMEDIATE_DIR)/security.md
	@echo "🔧 Generating feature tables and processing links..."
	$(PYTHON) scripts/tool_feat_table.py $(INTERMEDIATE_DIR) & \
	$(PYTHON) scripts/kv_store_feat_table.py $(INTERMEDIATE_DIR) & \
	$(PYTHON) scripts/partner_pkg_table.py $(INTERMEDIATE_DIR) & \
	$(PYTHON) scripts/resolve_local_links.py $(INTERMEDIATE_DIR)/langserve.md https://github.com/langchain-ai/langserve/tree/main/ & \
	wait
	@echo "✅ Files generated"

copy-infra:
	@echo "📂 Copying infrastructure files..."
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
	@echo "✅ Infrastructure files copied"

render:
	@echo "📓 Converting notebooks (this may take a while)..."
	$(PYTHON) scripts/notebook_convert.py $(INTERMEDIATE_DIR) $(OUTPUT_NEW_DOCS_DIR)
	@echo "✅ Notebooks converted"

md-sync:
	@echo "📝 Syncing markdown files..."
	rsync -avmq --include="*/" --include="*.mdx" --include="*.md" --include="*.png" --include="*/_category_.yml" --exclude="*" $(INTERMEDIATE_DIR)/ $(OUTPUT_NEW_DOCS_DIR)
	@echo "✅ Markdown files synced"

append-related:
	@echo "🔗 Appending related links..."
	$(PYTHON) scripts/append_related_links.py $(OUTPUT_NEW_DOCS_DIR)
	@echo "✅ Related links appended"

generate-references:
	$(PYTHON) scripts/generate_api_reference_links.py --docs_dir $(OUTPUT_NEW_DOCS_DIR)

update-md: generate-files md-sync

build: install-py-deps generate-files copy-infra render md-sync append-related
	@echo ""
	@echo "🎉 Documentation build complete!"
	@echo "📖 To view locally, run: cd docs && make start"
	@echo ""

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
	@echo "🚀 Starting documentation server on port $(PORT)..."
	@echo "📖 Installing Node.js dependencies..."
	cd $(OUTPUT_NEW_DIR) && yarn install --silent
	@echo "🌐 Starting server at http://localhost:$(PORT)"
	@echo "Press Ctrl+C to stop the server"
	cd $(OUTPUT_NEW_DIR) && yarn start --port=$(PORT)
