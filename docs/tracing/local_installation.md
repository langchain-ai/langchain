# Locally Hosted Setup

This page contains instructions for installing and then setting up the environment to use the locally hosted version of tracing.

## Installation

1. Ensure you have Docker installed (see [Get Docker](https://docs.docker.com/get-docker/)) and that it’s running.
2. Install the latest version of `langchain`: `pip install langchain` or `pip install langchain -U` to upgrade your
   existing version.
3. Run `langchain-server`
    1. This will spin up the server in the terminal.
    2. Once you see the terminal
       output `langchain-langchain-frontend-1 | ➜ Local: [http://localhost:4173/](http://localhost:4173/)`, navigate
       to [http://localhost:4173/](http://localhost:4173/)

4. You should see a page with your tracing sessions. See the overview page for a walkthrough of the UI.

5. Currently, trace data is not guaranteed to be persisted between runs of `langchain-server`. If you want to
       persist your data, you can mount a volume to the Docker container. See the [Docker docs](https://docs.docker.com/storage/volumes/) for more info.
6. To stop the server, press `Ctrl+C` in the terminal where you ran `langchain-server`.


## Environment Setup

After installation, you must now set up your environment to use tracing.

This can be done by setting an environment variable in your terminal by running `export LANGCHAIN_HANDLER=langchain`.

You can also do this by adding the below snippet to the top of every script. **IMPORTANT:** this must go at the VERY TOP of your script, before you import anything from `langchain`. 

```python
import os
os.environ["LANGCHAIN_HANDLER"] = "langchain"
```

