# Using Docker

To quickly get started, run the command `make docker`.

If docker is installed the Makefile will export extra targets in the fomrat `docker.*` to build and run the docker image. Type `make` for a list of available tasks.

There is a basic `docker-compose.yml` in the docker directory.

## Building the development image

Using `make docker` will build the dev image if it does not exist, then drops
you inside the container with the langchain environment available in the shell.

### Customizing the image and installed dependencies

The image is built with a default python version and all extras and dev
dependencies. It can be customized by changing the variables in the [.env](/docker/.env)
file. 

If you don't need all the `extra` dependencies a slimmer image can be obtained by 
commenting out `POETRY_EXTRA_PACKAGES` in the [.env](docker/.env) file.

### Image caching

The Dockerfile is optimized to cache the poetry install step. A rebuild is triggered when there a change to the source code.

## Example Usage

All commands from langchain's python environment are available by default in the container.

A few examples:
```bash
# run jupyter notebook
docker run --rm -it IMG jupyter notebook

# run ipython
docker run --rm -it IMG ipython

# start web server
docker run --rm -p 8888:8888 IMG python -m http.server 8888
```

## Testing / Linting

Tests and lints are run using your local source directory that is mounted on the volume /src.

Run unit tests in the container with `make docker.test`.

Run the linting and formatting checks with `make docker.lint`.

Note: this task can run in parallel using `make -j4 docker.lint`.


