## Using Docker

To quickly get started, run the command `make docker`.

If docker is installed the Makefile will export extra targets in the fomrat `docker.*` to build and run the docker image. Type `make` for a list of common tasks.

### Building the development image

- use `make docker.run` will build the dev image if it does not exist.
- `make docker.build` 

#### Customizing the image and installed dependencies

The image is built with a default python version and all extras and dev
dependencies. It can be customized by changing the variables in the [.env](/docker/.env)
file. 

If you don't need all the `extra` dependencies a slimmer image can be obtained by 
commenting out `POETRY_EXTRA_PACKAGES` in the [.env](docker/.env) file.

#### Image caching

The Dockerfile is optimized to cache the poetry install step. A rebuild is triggered when there a change to the source code.

### Example Usage

All commands that in the python env are available by default in the container.

A few examples:
```bash
# run jupyter notebook
docker run --rm -it IMG jupyter notebook

# run ipython
docker run --rm -it IMG ipython

# start web server
docker run --rm -p 8888:8888 IMG python -m http.server 8888
```
