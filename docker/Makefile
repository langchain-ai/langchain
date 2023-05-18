#do not call this makefile it is included in the main Makefile
.PHONY: docker docker.jupyter docker.run docker.force_build docker.clean

# read python version from .env file ignoring comments
PYTHON_VERSION := $(shell grep PYTHON_VERSION docker/.env | cut -d '=' -f2)
POETRY_EXTRA_PACKAGES := $(shell grep '^[^#]*POETRY_EXTRA_PACKAGES' docker/.env | cut -d '=' -f2)


DOCKER_SRC := $(shell find docker -type f)
DOCKER_IMAGE_NAME = langchain/dev

# SRC is all files matched by the git ls-files command
SRC := $(shell git ls-files -- '*' ':!:docker/*')

# set DOCKER_BUILD_PROGRESS=plain to see detailed build progress
DOCKER_BUILD_PROGRESS ?= auto

# extra message to show when entering the docker container
DOCKER_MOTD := docker/assets/etc/motd

ROOTDIR := $(shell git rev-parse --show-toplevel)

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
			--progress=$(DOCKER_BUILD_PROGRESS) \
			$(BUILD_FLAGS) -f docker/Dockerfile -t $(DOCKER_IMAGE_NAME):$(GIT_HASH) .
else
	docker build --build-arg PYTHON_VERSION=$(PYTHON_VERSION) \
			--build-arg POETRY_EXTRA_PACKAGES=$(POETRY_EXTRA_PACKAGES) \
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
