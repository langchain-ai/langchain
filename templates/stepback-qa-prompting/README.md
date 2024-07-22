# stepback-qa-prompting

This template replicates the "Step-Back" prompting technique that improves performance on complex questions by first asking a "step back" question. 

This technique can be combined with regular question-answering applications by doing retrieval on both the original and step-back question.

Read more about this in the paper [here](https://arxiv.org/abs/2310.06117) and an excellent blog post by Cobus Greyling [here](https://cobusgreyling.medium.com/a-new-prompt-engineering-technique-has-been-introduced-called-step-back-prompting-b00e8954cacb)

We will modify the prompts slightly to work better with chat models in this template.

## Environment Setup

Set the `OPENAI_API_KEY` environment variable to access the OpenAI models.

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package stepback-qa-prompting
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add stepback-qa-prompting
```

And add the following code to your `server.py` file:
```python
from stepback_qa_prompting.chain import chain as stepback_qa_prompting_chain

add_routes(app, stepback_qa_prompting_chain, path="/stepback-qa-prompting")
```

(Optional) Let's now configure LangSmith. 
LangSmith will help us trace, monitor and debug LangChain applications. 
You can sign up for LangSmith [here](https://smith.langchain.com/). 
If you don't have access, you can skip this section

```shell
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<your-api-key>
export LANGCHAIN_PROJECT=<your-project>  # if not specified, defaults to "default"
```

If you are inside this directory, then you can spin up a LangServe instance directly by:

```shell
langchain serve
```

This will start the FastAPI app with a server running locally at 
[http://localhost:8000](http://localhost:8000)

We can see all templates at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
We can access the playground at [http://127.0.0.1:8000/stepback-qa-prompting/playground](http://127.0.0.1:8000/stepback-qa-prompting/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/stepback-qa-prompting")
```