# ____project_name

<!--- This is a LangChain project bootstrapped by [LangChain CLI](https://github.com/langchain-ai/langchain). --->

## Customise

To customise this project, edit the following files:

- `____project_name_identifier/chain.py` contains an example chain, which you can edit to suit your needs.
- `____project_name_identifier/server.py` contains a FastAPI app that serves that chain using `langserve`. You can edit this to add more endpoints or customise your server.
- `tests/test_chain.py` contains tests for the chain. You can edit this to add more tests.
- `pyproject.toml` contains the project metadata, including the project name, version, and dependencies. You can edit this to add more dependencies or customise your project metadata.

## Install dependencies

If using poetry:

```bash
poetry install
```

If using vanilla pip:

```bash
pip install .
```

## Usage

To run the project locally, run

```
make start
```

This will launch a webserver on port 8001.

Or via docker compose (does not use hot reload by default):

```
docker compose up
```

## Deploy

To deploy the project, first build the docker image:

```
docker build . -t ____project_name_identifier:latest
```

Then run the image:

```
docker run -p 8001:8001 -e PORT=8001 ____project_name_identifier:latest
```

Don't forget to add any needed environment variables!

## Deploy to GCP

You can deploy to GCP Cloud Run using the following command:

First edit `.env.gcp.yaml` file with any environment variables you need. Then run:

```
make deploy_gcp
```

## Contributing

For information on how to set up your dev environment and contribute, see [here](.github/CONTRIBUTING.md).
