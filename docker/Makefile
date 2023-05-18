#do not call this makefile it is included in the main Makefile
.PHONY: docker docker.jupyter docker.run docker.force_build

# read python version from .env file
PYTHON_VERSION := $(shell grep PYTHON_VERSION docker/.env | cut -d '=' -f2)


DOCKER_SRC := $(shell find docker -type f -not -name .env)
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
	@docker run --rm -it -u lchain -v $(ROOTDIR):/src  $(DOCKER_IMAGE_NAME):$(GIT_HASH)
	@# $(local source mounted at $(ROOTDIR) at /src)

docker.jupyter: docker.build
	@docker run --rm -it -v $(ROOTDIR):/src  $(DOCKER_IMAGE_NAME):$(GIT_HASH) jupyter notebook

docker.build: $(SRC) $(DOCKER_SRC) $(DOCKER_MOTD)
ifdef $(DOCKER_BUILDKIT)
	@docker buildx build --build-arg PYTHON_VERSION=$(PYTHON_VERSION) \
			--progress=$(DOCKER_BUILD_PROGRESS) \
			-f docker/Dockerfile -t $(DOCKER_IMAGE_NAME):$(GIT_HASH) .
else
	@docker build --build-arg PYTHON_VERSION=$(PYTHON_VERSION) \
			-f docker/Dockerfile -t $(DOCKER_IMAGE_NAME):$(GIT_HASH) .
endif
	@docker tag $(DOCKER_IMAGE_NAME):$(GIT_HASH) $(DOCKER_IMAGE_NAME):latest
	@touch $@ # this avoids docker rebuilds the build dependencies have not changed
	@		  # you can remove the file docker_build to force a rebuild

docker.force_build: $(DOCKER_SRC)
	@docker build --no-cache -f docker/Dockerfile -t $(DOCKER_IMAGE_NAME):$(GIT_HASH) .

