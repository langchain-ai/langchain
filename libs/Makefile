# Makefile for libs/ directory
# Contains targets that operate across multiple packages

LANGCHAIN_DIRS = core text-splitters langchain langchain_v1 model-profiles

.PHONY: lock check-lock

# Regenerate lockfiles for all core packages
lock:
	@for dir in $(LANGCHAIN_DIRS); do \
		echo "=== Locking $$dir ==="; \
		(cd $$dir && uv lock); \
	done

# Verify all lockfiles are up-to-date
check-lock:
	@for dir in $(LANGCHAIN_DIRS); do \
		echo "=== Checking $$dir ==="; \
		(cd $$dir && uv lock --check) || exit 1; \
	done
