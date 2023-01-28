# LangChain Tracing Instructions

## Locally Hosted Setup

1. Ensure you have Docker installed (see [Get Docker](https://docs.docker.com/get-docker/)) and that it’s running.
2. Install the latest version of `langchain`: `pip install langchain` or `pip install langchain -U` to upgrade your
   existing version.
3. Run `langchain-server`
    1. This will spin up the server in the terminal.
    2. Once you see the terminal
       output `langchain-langchain-frontend-1 | ➜ Local: [http://localhost:4173/](http://localhost:4173/)`, navigate
       to [http://localhost:4173/](http://localhost:4173/)

4. You should see a page with your tracing sessions. An initial one "default" should already be created for you. A session is just a way to group traces together. If you click on a session, it will take you to a page with no recorded traces that says "No Runs." You can create a new session with the
   new session form. Now it’s time to get some data into the system.
5. How to log traces:
    1. Easiest way is to add this to the top of every script. **IMPORTANT:** this must go at the VERY TOP of your script, before you import anything from `langchain`. You can also set this environment variable in your terminal by running `export LANGCHAIN_HANDLER=langchain`.

        ```python
        import os
        os.environ["LANGCHAIN_HANDLER"] = "langchain"
        ```

    2. Full example

        ```python
        import os
        os.environ["LANGCHAIN_HANDLER"] = "langchain"
        
        from langchain import LLMMathChain
        from langchain.agents import Tool, initialize_agent
        from langchain.llms import OpenAI
        
        llm = OpenAI(temperature=0)
        llm_math_chain = LLMMathChain(llm=llm, verbose=True)
        tools = [
            Tool(
                name="Calculator",
                func=llm_math_chain.run,
                description="useful for when you need to answer questions about math",
            ),
        ]
        
        agent = initialize_agent(
            tools, llm, agent="zero-shot-react-description", verbose=True
        )
        
        agent.run("what is 3**.13")
        ```

6. You should see traces show up in the UI for a particular session. You can toggle to expand the traces and click
   “Explore” to view a particular run in more detail. There is a button that allows you to copy the inputs/outputs of each run to the clipboard. You can delete a top-level run by clicking the red “X”, which will
   pop up a confirmation window.
7. Options:
    1. To initially record traces to a session other than `"default"`, you can set the `LANGCHAIN_SESSION` environment
       variable to the name of the session you want to record to:

        ```python
        import os
        os.environ["LANGCHAIN_HANDLER"] = "langchain"
        os.environ["LANGCHAIN_SESSION"] = "my_session" # Make sure this session actually exists. You can create a new session in the UI.
        ```
    2. To switch sessions mid-script or mid-notebook, do NOT set the `LANGCHAIN_SESSION` environment variable. Instead: `langchain.set_tracing_callback_manager(session_name="my_session")`
8. Notes:
    1. Currently, trace data is not guaranteed to be persisted between runs of `langchain-server`. If you want to
       persist your data, you can mount a volume to the Docker container. See the [Docker
       docs](https://docs.docker.com/storage/volumes/) for more info.
2. To stop the server, press `Ctrl+C` in the terminal where you ran `langchain-server`.

## Hosted Setup

We offer a hosted version of tracing at https://langchainplus.vercel.app/. You can use this to view traces from your run without having to run the server locally.

Note: we are currently only offering this to a limited number of users. The hosted platform is VERY alpha, in active development, and data might be dropped at any time. Don't depend on data being persisted in the system long term and don't log traces that may contain sensitive information. If you're interested in using the hosted platform, please fill out the form [here](https://forms.gle/tRCEMSeopZf6TE3b6).

1. Login to the system and click "API Key" in the top right corner. Generate a new key and keep it safe. You will need it to authenticate with the system.
2. Follow the instructions from the previous section (4. onwards) to log and view traces. You will need to set two additional environment variables:
    1. `LANGCHAIN_ENDPOINT` = "https://langchain-api-gateway-57eoxz8z.uc.gateway.dev"
    2. `LANGCHAIN_API_KEY` - set this to the API key you generated in step 1.
        ```python
        import os
        os.environ["LANGCHAIN_HANDLER"] = "langchain"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://langchain-api-gateway-57eoxz8z.uc.gateway.dev"
        os.environ["LANGCHAIN_API_KEY"] = "my_api_key"  # Don't commit this to your repo! Better to set it in your terminal.
        ```
