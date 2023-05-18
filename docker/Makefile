#do not call this makefile it is included in the main Makefile
.PHONY: docker docker.jupyter docker.run docker.force_build docker.clean \
	docker.test docker.lint docker.lint.mypy docker.lint.black \
	docker.lint.isort docker.lint.flake

# read python version from .env file ignoring comments
PYTHON_VERSION := $(shell grep PYTHON_VERSION docker/.env | cut -d '=' -f2)
POETRY_EXTRA_PACKAGES := $(shell grep '^[^#]*POETRY_EXTRA_PACKAGES' docker/.env | cut -d '=' -f2)
POETRY_DEPENDENCIES := $(shell grep 'POETRY_DEPENDENCIES' docker/.env | cut -d '=' -f2)


DOCKER_SRC := $(shell find docker -type f)
DOCKER_IMAGE_NAME = langchain/dev

# SRC is all files matched by the git ls-files command
SRC := $(shell git ls-files -- '*' ':!:docker/*')

# set DOCKER_BUILD_PROGRESS=plain to see detailed build progress
DOCKER_BUILD_PROGRESS ?= auto

# extra message to show when entering the docker container
DOCKER_MOTD := docker/assets/etc/motd

ROOTDIR := $(shell git rev-parse --show-toplevel)

DOCKER_LINT_CMD = docker run --rm -i -u lchain -v $(ROOTDIR):/src  $(DOCKER_IMAGE_NAME):$(GIT_HASH)

docker: docker.run

docker.run: docker.build
	@echo "Docker image: $(DOCKER_IMAGE_NAME):$(GIT_HASH)"
	docker run --rm -it -u lchain -v $(ROOTDIR):/src  $(DOCKER_IMAGE_NAME):$(GIT_HASH)

docker.jupyter: docker.build
	docker run --rm -it -v $(ROOTDIR):/src  $(DOCKER_IMAGE_NAME):$(GIT_HASH) jupyter notebook

docker.build: $(SRC) $(DOCKER_SRC) $(DOCKER_MOTD)
ifdef $(DOCKER_BUILDKIT)
	docker buildx build --build-arg PYTHON_VERSION=$(PYTHON_VERSION) \
			--build-arg POETRY_EXTRA_PACKAGES=$(POETRY_EXTRA_PACKAGES) \
			--build-arg POETRY_DEPENDENCIES=$(POETRY_DEPENDENCIES) \
			--progress=$(DOCKER_BUILD_PROGRESS) \
			$(BUILD_FLAGS) -f docker/Dockerfile -t $(DOCKER_IMAGE_NAME):$(GIT_HASH) .
else
	docker build --build-arg PYTHON_VERSION=$(PYTHON_VERSION) \
			--build-arg POETRY_EXTRA_PACKAGES=$(POETRY_EXTRA_PACKAGES) \
			--build-arg POETRY_DEPENDENCIES=$(POETRY_DEPENDENCIES) \
			$(BUILD_FLAGS) -f docker/Dockerfile -t $(DOCKER_IMAGE_NAME):$(GIT_HASH) .
endif
	docker tag $(DOCKER_IMAGE_NAME):$(GIT_HASH) $(DOCKER_IMAGE_NAME):latest
	@touch $@ # this prevents docker from rebuilding dependencies that have not 
	@         #  changed. Remove the file `docker/docker.build` to force a rebuild.

docker.force_build: $(DOCKER_SRC)
	@rm -f docker.build
	@$(MAKE) docker.build BUILD_FLAGS=--no-cache

docker.clean:
	docker rmi $(DOCKER_IMAGE_NAME):$(GIT_HASH) $(DOCKER_IMAGE_NAME):latest

docker.test: docker.build
	docker run --rm -it -u lchain -v $(ROOTDIR):/src  $(DOCKER_IMAGE_NAME):$(GIT_HASH) \
		pytest /src/tests/unit_tests

# this assumes that the docker image has been built 
docker.lint: docker.lint.mypy docker.lint.black docker.lint.isort \
	docker.lint.flake

# these can run in parallel with -j[njobs]
docker.lint.mypy:
	@$(DOCKER_LINT_CMD) mypy /src
	@printf "\t%s\n" "mypy ... "

docker.lint.black:
	@$(DOCKER_LINT_CMD) black /src --check
	@printf "\t%s\n" "black ... "

docker.lint.isort:
	@$(DOCKER_LINT_CMD) isort /src --check
	@printf "\t%s\n" "isort ... "

docker.lint.flake:
	@$(DOCKER_LINT_CMD) flake8 /src
	@printf "\t%s\n" "flake8 ... "
