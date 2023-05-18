# vim: ft=dockerfile
#
# see also: https://github.com/python-poetry/poetry/discussions/1879
#   - with https://github.com/bneijt/poetry-lock-docker
# see https://github.com/thehale/docker-python-poetry
# see https://github.com/max-pfeiffer/uvicorn-poetry

# use by default the slim version of python
ARG PYTHON_IMAGE_TAG=slim 
ARG PYTHON_VERSION=${PYTHON_VERSION:-3.11.2}

####################
# Base Environment
####################
FROM python:$PYTHON_VERSION-$PYTHON_IMAGE_TAG AS lchain-base

ARG UID=1000
ARG USERNAME=lchain

ENV USERNAME=$USERNAME

RUN groupadd -g ${UID} $USERNAME
RUN useradd -l -m -u ${UID} -g ${UID} $USERNAME

# used for mounting source code
RUN mkdir /src
VOLUME /src


#######################
## Poetry Builder Image
#######################
FROM lchain-base AS lchain-base-builder

ARG POETRY_EXTRA_PACKAGES=${POETRY_EXTRA_PACKAGES}
ENV POETRY_EXTRA_PACKAGES=$POETRY_EXTRA_PACKAGES

ENV HOME=/root
ENV POETRY_HOME=/root/.poetry
ENV POETRY_VIRTUALENVS_IN_PROJECT=false
ENV POETRY_NO_INTERACTION=1
ENV CACHE_DIR=$HOME/.cache
ENV POETRY_CACHE_DIR=$CACHE_DIR/pypoetry
ENV PATH="$POETRY_HOME/bin:$PATH"

WORKDIR /root

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    git \
    curl

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN mkdir -p $CACHE_DIR

## setup poetry
RUN curl -sSL -o $CACHE_DIR/pypoetry-installer.py https://install.python-poetry.org/
RUN python3 $CACHE_DIR/pypoetry-installer.py


# # Copy poetry files
COPY poetry.* pyproject.toml ./

RUN mkdir /pip-prefix

RUN poetry export $POETRY_EXTRA_PACKAGES --with dev -f requirements.txt --output requirements.txt --without-hashes && \
    pip install --no-cache-dir --disable-pip-version-check --prefix /pip-prefix -r requirements.txt


# add custom motd message
COPY docker/assets/etc/motd /tmp/motd
RUN cat /tmp/motd > /etc/motd

RUN printf "\n%s\n%s\n" "$(poetry version)" "$(python --version)" >> /etc/motd

###################
## Runtime Image
###################
FROM lchain-base AS lchain

#jupyter port
EXPOSE 8888

COPY docker/assets/entry.sh /entry
RUN chmod +x /entry

COPY --from=lchain-base-builder /etc/motd /etc/motd
COPY --from=lchain-base-builder /usr/bin/git /usr/bin/git

USER ${USERNAME:-lchain}
ENV HOME /home/$USERNAME
WORKDIR /home/$USERNAME

COPY --chown=lchain:lchain --from=lchain-base-builder /pip-prefix $HOME/.local/

COPY . .

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN pip install --no-deps --disable-pip-version-check --no-cache-dir -e .


entrypoint ["/entry"]
