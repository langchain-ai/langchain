# Makefile for libs/partners/ directory
# Contains targets that operate across all partner packages

PARTNER_DIRS = anthropic chroma deepseek exa fireworks groq huggingface mistralai nomic ollama openai openrouter perplexity qdrant xai

.PHONY: lock check-lock

# Regenerate lockfiles for all partner packages
lock:
	@for dir in $(PARTNER_DIRS); do \
		echo "=== Locking $$dir ==="; \
		(cd $$dir && uv lock); \
	done

# Verify all lockfiles are up-to-date
check-lock:
	@for dir in $(PARTNER_DIRS); do \
		echo "=== Checking $$dir ==="; \
		(cd $$dir && uv lock --check) || exit 1; \
	done
