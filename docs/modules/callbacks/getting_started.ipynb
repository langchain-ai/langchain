{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "23234b50-e6c6-4c87-9f97-259c15f36894",
   "metadata": {
    "tags": []
   },
   "source": [
    "# Callbacks"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "29dd6333-307c-43df-b848-65001c01733b",
   "metadata": {},
   "source": [
    "LangChain provides a callbacks system that allows you to hook into the various stages of your LLM application. This is useful for logging, [monitoring](https://python.langchain.com/en/latest/tracing.html), [streaming](https://python.langchain.com/en/latest/modules/models/llms/examples/streaming_llm.html), and other tasks.\n",
    "\n",
    "You can subscribe to these events by using the `callbacks` argument available throughout the API. This argument is list of handler objects, which are expected to implement one or more of the methods described below in more detail. There are two main callbacks mechanisms:\n",
    "\n",
    "* *Constructor callbacks* will be used for all calls made on that object, and will be scoped to that object only, i.e. if you pass a handler to the `LLMChain` constructor, it will not be used by the model attached to that chain. \n",
    "* *Request callbacks* will be used for that specific request only, and all sub-requests that it contains (eg. a call to an `LLMChain` triggers a call to a Model, which uses the same handler passed through). These are explicitly passed through.\n",
    "\n",
    "\n",
    "**Advanced:** When you create a custom chain you can easily set it up to use the same callback system as all the built-in chains. \n",
    "`_call`, `_generate`, `_run`, and equivalent async methods on Chains / LLMs / Chat Models / Agents / Tools now receive a 2nd argument called `run_manager` which is bound to that run, and contains the logging methods that can be used by that object (i.e. `on_llm_new_token`). This is useful when constructing a custom chain. See this guide for more information on how to [create custom chains and use callbacks inside them.](https://python.langchain.com/en/latest/modules/chains/generic/custom_chain.html)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2b6d7dba-cd20-472a-ae05-f68675cc9ea4",
   "metadata": {},
   "source": [
    "`CallbackHandlers` are objects that implement the `CallbackHandler` interface, which has a method for each event that can be subscribed to. The `CallbackManager` will call the appropriate method on each handler when the event is triggered."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e4592215-6604-47e2-89ff-5db3af6d1e40",
   "metadata": {
    "tags": []
   },
   "source": [
    "```python\n",
    "class BaseCallbackHandler:\n",
    "    \"\"\"Base callback handler that can be used to handle callbacks from langchain.\"\"\"\n",
    "\n",
    "    def on_llm_start(\n",
    "        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any\n",
    "    ) -> Any:\n",
    "        \"\"\"Run when LLM starts running.\"\"\"\n",
    "\n",
    "    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:\n",
    "        \"\"\"Run on new LLM token. Only available when streaming is enabled.\"\"\"\n",
    "\n",
    "    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:\n",
    "        \"\"\"Run when LLM ends running.\"\"\"\n",
    "\n",
    "    def on_llm_error(\n",
    "        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any\n",
    "    ) -> Any:\n",
    "        \"\"\"Run when LLM errors.\"\"\"\n",
    "\n",
    "    def on_chain_start(\n",
    "        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any\n",
    "    ) -> Any:\n",
    "        \"\"\"Run when chain starts running.\"\"\"\n",
    "\n",
    "    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:\n",
    "        \"\"\"Run when chain ends running.\"\"\"\n",
    "\n",
    "    def on_chain_error(\n",
    "        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any\n",
    "    ) -> Any:\n",
    "        \"\"\"Run when chain errors.\"\"\"\n",
    "\n",
    "    def on_tool_start(\n",
    "        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any\n",
    "    ) -> Any:\n",
    "        \"\"\"Run when tool starts running.\"\"\"\n",
    "\n",
    "    def on_tool_end(self, output: str, **kwargs: Any) -> Any:\n",
    "        \"\"\"Run when tool ends running.\"\"\"\n",
    "\n",
    "    def on_tool_error(\n",
    "        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any\n",
    "    ) -> Any:\n",
    "        \"\"\"Run when tool errors.\"\"\"\n",
    "\n",
    "    def on_text(self, text: str, **kwargs: Any) -> Any:\n",
    "        \"\"\"Run on arbitrary text.\"\"\"\n",
    "\n",
    "    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:\n",
    "        \"\"\"Run on agent action.\"\"\"\n",
    "\n",
    "    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:\n",
    "        \"\"\"Run on agent end.\"\"\"\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cbccd7d1",
   "metadata": {},
   "source": [
    "## How to use callbacks\n",
    "\n",
    "The `callbacks` argument is available on most objects throughout the API (Chains, Models, Tools, Agents, etc.) in two different places:\n",
    "\n",
    "- **Constructor callbacks**: defined in the constructor, eg. `LLMChain(callbacks=[handler])`, which will be used for all calls made on that object, and will be scoped to that object only, eg. if you pass a handler to the `LLMChain` constructor, it will not be used by the Model attached to that chain.\n",
    "- **Request callbacks**: defined in the `call()`/`run()`/`apply()` methods used for issuing a request, eg. `chain.call(inputs, callbacks=[handler])`, which will be used for that specific request only, and all sub-requests that it contains (eg. a call to an LLMChain triggers a call to a Model, which uses the same handler passed in the `call()` method).\n",
    "\n",
    "The `verbose` argument is available on most objects throughout the API (Chains, Models, Tools, Agents, etc.) as a constructor argument, eg. `LLMChain(verbose=True)`, and it is equivalent to passing a `ConsoleCallbackHandler` to the `callbacks` argument of that object and all child objects. This is useful for debugging, as it will log all events to the console.\n",
    "\n",
    "### When do you want to use each of these?\n",
    "\n",
    "- Constructor callbacks are most useful for use cases such as logging, monitoring, etc., which are _not specific to a single request_, but rather to the entire chain. For example, if you want to log all the requests made to an LLMChain, you would pass a handler to the constructor.\n",
    "- Request callbacks are most useful for use cases such as streaming, where you want to stream the output of a single request to a specific websocket connection, or other similar use cases. For example, if you want to stream the output of a single request to a websocket, you would pass a handler to the `call()` method"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d3bf3304-43fb-47ad-ae50-0637a17018a2",
   "metadata": {},
   "source": [
    "## Using an existing handler\n",
    "\n",
    "LangChain provides a few built-in handlers that you can use to get started. These are available in the `langchain/callbacks` module. The most basic handler is the `StdOutCallbackHandler`, which simply logs all events to `stdout`. In the future we will add more default handlers to the library. \n",
    "\n",
    "**Note** when the `verbose` flag on the object is set to true, the `StdOutCallbackHandler` will be invoked even without being explicitly passed in."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "80532dfc-d687-4147-a0c9-1f90cc3e868c",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "\n",
      "\u001b[1m> Entering new LLMChain chain...\u001b[0m\n",
      "Prompt after formatting:\n",
      "\u001b[32;1m\u001b[1;3m1 + 2 = \u001b[0m\n",
      "\n",
      "\u001b[1m> Finished chain.\u001b[0m\n",
      "\n",
      "\n",
      "\u001b[1m> Entering new LLMChain chain...\u001b[0m\n",
      "Prompt after formatting:\n",
      "\u001b[32;1m\u001b[1;3m1 + 2 = \u001b[0m\n",
      "\n",
      "\u001b[1m> Finished chain.\u001b[0m\n",
      "\n",
      "\n",
      "\u001b[1m> Entering new LLMChain chain...\u001b[0m\n",
      "Prompt after formatting:\n",
      "\u001b[32;1m\u001b[1;3m1 + 2 = \u001b[0m\n",
      "\n",
      "\u001b[1m> Finished chain.\u001b[0m\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "'\\n\\n3'"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from langchain.callbacks import StdOutCallbackHandler\n",
    "from langchain.chains import LLMChain\n",
    "from langchain.llms import OpenAI\n",
    "from langchain.prompts import PromptTemplate\n",
    "\n",
    "handler = StdOutCallbackHandler()\n",
    "llm = OpenAI()\n",
    "prompt = PromptTemplate.from_template(\"1 + {number} = \")\n",
    "\n",
    "# First, let's explicitly set the StdOutCallbackHandler in `callbacks`\n",
    "chain = LLMChain(llm=llm, prompt=prompt, callbacks=[handler])\n",
    "chain.run(number=2)\n",
    "\n",
    "# Then, let's use the `verbose` flag to achieve the same result\n",
    "chain = LLMChain(llm=llm, prompt=prompt, verbose=True)\n",
    "chain.run(number=2)\n",
    "\n",
    "# Finally, let's use the request `callbacks` to achieve the same result\n",
    "chain = LLMChain(llm=llm, prompt=prompt)\n",
    "chain.run(number=2, callbacks=[handler])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "389c8448-5283-49e3-8c04-dbe1522e202c",
   "metadata": {},
   "source": [
    "## Creating a custom handler\n",
    "\n",
    "You can create a custom handler to set on the object as well. In the example below, we'll implement streaming with a custom handler."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "1b2e6588-0681-4cab-937a-7cc4790cea9a",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "My custom handler, token: \n",
      "My custom handler, token: Why\n",
      "My custom handler, token:  did\n",
      "My custom handler, token:  the\n",
      "My custom handler, token:  tomato\n",
      "My custom handler, token:  turn\n",
      "My custom handler, token:  red\n",
      "My custom handler, token: ?\n",
      "My custom handler, token:  Because\n",
      "My custom handler, token:  it\n",
      "My custom handler, token:  saw\n",
      "My custom handler, token:  the\n",
      "My custom handler, token:  salad\n",
      "My custom handler, token:  dressing\n",
      "My custom handler, token: !\n",
      "My custom handler, token: \n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "AIMessage(content='Why did the tomato turn red? Because it saw the salad dressing!', additional_kwargs={})"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from langchain.callbacks.base import BaseCallbackHandler\n",
    "from langchain.chat_models import ChatOpenAI\n",
    "from langchain.schema import HumanMessage\n",
    "\n",
    "class MyCustomHandler(BaseCallbackHandler):\n",
    "    def on_llm_new_token(self, token: str, **kwargs) -> None:\n",
    "        print(f\"My custom handler, token: {token}\")\n",
    "\n",
    "# To enable streaming, we pass in `streaming=True` to the ChatModel constructor\n",
    "# Additionally, we pass in a list with our custom handler\n",
    "chat = ChatOpenAI(max_tokens=25, streaming=True, callbacks=[MyCustomHandler()])\n",
    "\n",
    "chat([HumanMessage(content=\"Tell me a joke\")])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bc9785fa-4f71-4797-91a3-4fe7e57d0429",
   "metadata": {
    "tags": []
   },
   "source": [
    "## Async Callbacks\n",
    "\n",
    "If you are planning to use the async API, it is recommended to use `AsyncCallbackHandler` to avoid blocking the runloop. \n",
    "\n",
    "**Advanced** if you use a sync `CallbackHandler` while using an async method to run your llm/chain/tool/agent, it will still work. However, under the hood, it will be called with [`run_in_executor`](https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.run_in_executor) which can cause issues if your `CallbackHandler` is not thread-safe."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "c702e0c9-a961-4897-90c1-cdd13b6f16b2",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "zzzz....\n",
      "Hi! I just woke up. Your llm is starting\n",
      "Sync handler being called in a `thread_pool_executor`: token: \n",
      "Sync handler being called in a `thread_pool_executor`: token: Why\n",
      "Sync handler being called in a `thread_pool_executor`: token:  don\n",
      "Sync handler being called in a `thread_pool_executor`: token: 't\n",
      "Sync handler being called in a `thread_pool_executor`: token:  scientists\n",
      "Sync handler being called in a `thread_pool_executor`: token:  trust\n",
      "Sync handler being called in a `thread_pool_executor`: token:  atoms\n",
      "Sync handler being called in a `thread_pool_executor`: token: ?\n",
      "\n",
      "\n",
      "Sync handler being called in a `thread_pool_executor`: token: Because\n",
      "Sync handler being called in a `thread_pool_executor`: token:  they\n",
      "Sync handler being called in a `thread_pool_executor`: token:  make\n",
      "Sync handler being called in a `thread_pool_executor`: token:  up\n",
      "Sync handler being called in a `thread_pool_executor`: token:  everything\n",
      "Sync handler being called in a `thread_pool_executor`: token: !\n",
      "Sync handler being called in a `thread_pool_executor`: token: \n",
      "zzzz....\n",
      "Hi! I just woke up. Your llm is ending\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "LLMResult(generations=[[ChatGeneration(text=\"Why don't scientists trust atoms?\\n\\nBecause they make up everything!\", generation_info=None, message=AIMessage(content=\"Why don't scientists trust atoms?\\n\\nBecause they make up everything!\", additional_kwargs={}))]], llm_output={'token_usage': {}, 'model_name': 'gpt-3.5-turbo'})"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import asyncio\n",
    "from typing import Any, Dict, List\n",
    "from langchain.schema import LLMResult\n",
    "from langchain.callbacks.base import AsyncCallbackHandler\n",
    "\n",
    "class MyCustomSyncHandler(BaseCallbackHandler):\n",
    "    def on_llm_new_token(self, token: str, **kwargs) -> None:\n",
    "        print(f\"Sync handler being called in a `thread_pool_executor`: token: {token}\")\n",
    "\n",
    "class MyCustomAsyncHandler(AsyncCallbackHandler):\n",
    "    \"\"\"Async callback handler that can be used to handle callbacks from langchain.\"\"\"\n",
    "\n",
    "    async def on_llm_start(\n",
    "        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any\n",
    "    ) -> None:\n",
    "        \"\"\"Run when chain starts running.\"\"\"\n",
    "        print(\"zzzz....\")\n",
    "        await asyncio.sleep(0.3)\n",
    "        class_name = serialized[\"name\"]\n",
    "        print(\"Hi! I just woke up. Your llm is starting\")\n",
    "\n",
    "    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:\n",
    "        \"\"\"Run when chain ends running.\"\"\"\n",
    "        print(\"zzzz....\")\n",
    "        await asyncio.sleep(0.3)\n",
    "        print(\"Hi! I just woke up. Your llm is ending\")\n",
    "\n",
    "# To enable streaming, we pass in `streaming=True` to the ChatModel constructor\n",
    "# Additionally, we pass in a list with our custom handler\n",
    "chat = ChatOpenAI(max_tokens=25, streaming=True, callbacks=[MyCustomSyncHandler(), MyCustomAsyncHandler()])\n",
    "\n",
    "await chat.agenerate([[HumanMessage(content=\"Tell me a joke\")]])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d26dbb34-fcc3-401c-a115-39c7620d2d65",
   "metadata": {},
   "source": [
    "## Using multiple handlers, passing in handlers\n",
    "\n",
    "In the previous examples, we passed in callback handlers upon creation of an object by using `callbacks=`. In this case, the callbacks will be scoped to that particular object. \n",
    "\n",
    "However, in many cases, it is advantageous to pass in handlers instead when running the object. When we pass through `CallbackHandlers` using the `callbacks` keyword arg when executing an run, those callbacks will be issued by all nested objects involved in the execution. For example, when a handler is passed through to an `Agent`, it will be used for all callbacks related to the agent and all the objects involved in the agent's execution, in this case, the `Tools`, `LLMChain`, and `LLM`.\n",
    "\n",
    "This prevents us from having to manually attach the handlers to each individual nested object."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "8eec8756-1828-45cb-9699-38ac8543a150",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "on_chain_start AgentExecutor\n",
      "on_chain_start LLMChain\n",
      "on_llm_start OpenAI\n",
      "on_llm_start (I'm the second handler!!) OpenAI\n",
      "on_new_token  I\n",
      "on_new_token  need\n",
      "on_new_token  to\n",
      "on_new_token  use\n",
      "on_new_token  a\n",
      "on_new_token  calculator\n",
      "on_new_token  to\n",
      "on_new_token  solve\n",
      "on_new_token  this\n",
      "on_new_token .\n",
      "on_new_token \n",
      "Action\n",
      "on_new_token :\n",
      "on_new_token  Calculator\n",
      "on_new_token \n",
      "Action\n",
      "on_new_token  Input\n",
      "on_new_token :\n",
      "on_new_token  2\n",
      "on_new_token ^\n",
      "on_new_token 0\n",
      "on_new_token .\n",
      "on_new_token 235\n",
      "on_new_token \n",
      "on_agent_action AgentAction(tool='Calculator', tool_input='2^0.235', log=' I need to use a calculator to solve this.\\nAction: Calculator\\nAction Input: 2^0.235')\n",
      "on_tool_start Calculator\n",
      "on_chain_start LLMMathChain\n",
      "on_chain_start LLMChain\n",
      "on_llm_start OpenAI\n",
      "on_llm_start (I'm the second handler!!) OpenAI\n",
      "on_new_token \n",
      "\n",
      "on_new_token ```text\n",
      "on_new_token \n",
      "\n",
      "on_new_token 2\n",
      "on_new_token **\n",
      "on_new_token 0\n",
      "on_new_token .\n",
      "on_new_token 235\n",
      "on_new_token \n",
      "\n",
      "on_new_token ```\n",
      "\n",
      "on_new_token ...\n",
      "on_new_token num\n",
      "on_new_token expr\n",
      "on_new_token .\n",
      "on_new_token evaluate\n",
      "on_new_token (\"\n",
      "on_new_token 2\n",
      "on_new_token **\n",
      "on_new_token 0\n",
      "on_new_token .\n",
      "on_new_token 235\n",
      "on_new_token \")\n",
      "on_new_token ...\n",
      "on_new_token \n",
      "\n",
      "on_new_token \n",
      "on_chain_start LLMChain\n",
      "on_llm_start OpenAI\n",
      "on_llm_start (I'm the second handler!!) OpenAI\n",
      "on_new_token  I\n",
      "on_new_token  now\n",
      "on_new_token  know\n",
      "on_new_token  the\n",
      "on_new_token  final\n",
      "on_new_token  answer\n",
      "on_new_token .\n",
      "on_new_token \n",
      "Final\n",
      "on_new_token  Answer\n",
      "on_new_token :\n",
      "on_new_token  1\n",
      "on_new_token .\n",
      "on_new_token 17\n",
      "on_new_token 690\n",
      "on_new_token 67\n",
      "on_new_token 372\n",
      "on_new_token 187\n",
      "on_new_token 674\n",
      "on_new_token \n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "'1.1769067372187674'"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from typing import Dict, Union, Any, List\n",
    "\n",
    "from langchain.callbacks.base import BaseCallbackHandler\n",
    "from langchain.schema import AgentAction\n",
    "from langchain.agents import AgentType, initialize_agent, load_tools\n",
    "from langchain.callbacks import tracing_enabled\n",
    "from langchain.llms import OpenAI\n",
    "\n",
    "# First, define custom callback handler implementations\n",
    "class MyCustomHandlerOne(BaseCallbackHandler):\n",
    "    def on_llm_start(\n",
    "        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any\n",
    "    ) -> Any:\n",
    "        print(f\"on_llm_start {serialized['name']}\")\n",
    "\n",
    "    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:\n",
    "        print(f\"on_new_token {token}\")\n",
    "\n",
    "    def on_llm_error(\n",
    "        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any\n",
    "    ) -> Any:\n",
    "        \"\"\"Run when LLM errors.\"\"\"\n",
    "\n",
    "    def on_chain_start(\n",
    "        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any\n",
    "    ) -> Any:\n",
    "        print(f\"on_chain_start {serialized['name']}\")\n",
    "\n",
    "    def on_tool_start(\n",
    "        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any\n",
    "    ) -> Any:\n",
    "        print(f\"on_tool_start {serialized['name']}\")\n",
    "\n",
    "    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:\n",
    "        print(f\"on_agent_action {action}\")\n",
    "\n",
    "class MyCustomHandlerTwo(BaseCallbackHandler):\n",
    "    def on_llm_start(\n",
    "        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any\n",
    "    ) -> Any:\n",
    "        print(f\"on_llm_start (I'm the second handler!!) {serialized['name']}\")\n",
    "\n",
    "# Instantiate the handlers\n",
    "handler1 = MyCustomHandlerOne()\n",
    "handler2 = MyCustomHandlerTwo()\n",
    "\n",
    "# Setup the agent. Only the `llm` will issue callbacks for handler2\n",
    "llm = OpenAI(temperature=0, streaming=True, callbacks=[handler2])\n",
    "tools = load_tools([\"llm-math\"], llm=llm)\n",
    "agent = initialize_agent(\n",
    "    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION\n",
    ")\n",
    "\n",
    "# Callbacks for handler1 will be issued by every object involved in the \n",
    "# Agent execution (llm, llmchain, tool, agent executor)\n",
    "agent.run(\"What is 2 raised to the 0.235 power?\", callbacks=[handler1])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "32b29135-f852-4492-88ed-547275c72c53",
   "metadata": {},
   "source": [
    "# Tracing and Token Counting"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fbb606d6-2863-46c5-8347-9f0bdb3805bb",
   "metadata": {},
   "source": [
    "Tracing and token counting are two capabilities we provide which are built on our callbacks mechanism."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f62cd10c-494c-47d6-aa98-6e926cb9c456",
   "metadata": {},
   "source": [
    "## Tracing"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d5a74b3f-3769-4a4f-99c7-b6a3b20a94e2",
   "metadata": {},
   "source": [
    "There are two recommended ways to trace your LangChains:\n",
    "\n",
    "1. Setting the `LANGCHAIN_TRACING` environment variable to `\"true\"`. \n",
    "2. Using a context manager `with tracing_enabled()` to trace a particular block of code.\n",
    "\n",
    "**Note** if the environment variable is set, all code will be traced, regardless of whether or not it's within the context manager."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "f164dfd5-d987-4b6a-a7c8-019c651ce47f",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "import os\n",
    "\n",
    "from langchain.agents import AgentType, initialize_agent, load_tools\n",
    "from langchain.callbacks import tracing_enabled\n",
    "from langchain.llms import OpenAI\n",
    "\n",
    "# To run the code, make sure to set OPENAI_API_KEY and SERPAPI_API_KEY\n",
    "llm = OpenAI(temperature=0)\n",
    "tools = load_tools([\"llm-math\", \"serpapi\"], llm=llm)\n",
    "agent = initialize_agent(\n",
    "    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True\n",
    ")\n",
    "\n",
    "questions = [\n",
    "    \"Who won the US Open men's final in 2019? What is his age raised to the 0.334 power?\",\n",
    "    \"Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?\",\n",
    "    \"Who won the most recent formula 1 grand prix? What is their age raised to the 0.23 power?\",\n",
    "    \"Who won the US Open women's final in 2019? What is her age raised to the 0.34 power?\",\n",
    "    \"Who is Beyonce's husband? What is his age raised to the 0.19 power?\",\n",
    "]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "6be7777e-ec1d-438f-ae33-3a93c45f808e",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "\n",
      "\u001b[1m> Entering new AgentExecutor chain...\u001b[0m\n",
      "\u001b[32;1m\u001b[1;3m I need to find out who won the US Open men's final in 2019 and then calculate his age raised to the 0.334 power.\n",
      "Action: Search\n",
      "Action Input: \"US Open men's final 2019 winner\"\u001b[0m\n",
      "Observation: \u001b[33;1m\u001b[1;3mRafael Nadal defeated Daniil Medvedev in the final, 7–5, 6–3, 5–7, 4–6, 6–4 to win the men's singles tennis title at the 2019 US Open. It was his fourth US ...\u001b[0m\n",
      "Thought:\u001b[32;1m\u001b[1;3m I need to find out the age of the winner\n",
      "Action: Search\n",
      "Action Input: \"Rafael Nadal age\"\u001b[0m\n",
      "Observation: \u001b[33;1m\u001b[1;3m36 years\u001b[0m\n",
      "Thought:\u001b[32;1m\u001b[1;3m I need to calculate the age raised to the 0.334 power\n",
      "Action: Calculator\n",
      "Action Input: 36^0.334\u001b[0m\n",
      "Observation: \u001b[36;1m\u001b[1;3mAnswer: 3.3098250249682484\u001b[0m\n",
      "Thought:\u001b[32;1m\u001b[1;3m I now know the final answer\n",
      "Final Answer: Rafael Nadal, aged 36, won the US Open men's final in 2019 and his age raised to the 0.334 power is 3.3098250249682484.\u001b[0m\n",
      "\n",
      "\u001b[1m> Finished chain.\u001b[0m\n",
      "\n",
      "\n",
      "\u001b[1m> Entering new AgentExecutor chain...\u001b[0m\n",
      "\u001b[32;1m\u001b[1;3m I need to find out who Olivia Wilde's boyfriend is and then calculate his age raised to the 0.23 power.\n",
      "Action: Search\n",
      "Action Input: \"Olivia Wilde boyfriend\"\u001b[0m\n",
      "Observation: \u001b[33;1m\u001b[1;3mSudeikis and Wilde's relationship ended in November 2020. Wilde was publicly served with court documents regarding child custody while she was presenting Don't Worry Darling at CinemaCon 2022. In January 2021, Wilde began dating singer Harry Styles after meeting during the filming of Don't Worry Darling.\u001b[0m\n",
      "Thought:\u001b[32;1m\u001b[1;3m I need to find out Harry Styles' age.\n",
      "Action: Search\n",
      "Action Input: \"Harry Styles age\"\u001b[0m\n",
      "Observation: \u001b[33;1m\u001b[1;3m29 years\u001b[0m\n",
      "Thought:\u001b[32;1m\u001b[1;3m I need to calculate 29 raised to the 0.23 power.\n",
      "Action: Calculator\n",
      "Action Input: 29^0.23\u001b[0m\n",
      "Observation: \u001b[36;1m\u001b[1;3mAnswer: 2.169459462491557\u001b[0m\n",
      "Thought:\u001b[32;1m\u001b[1;3m I now know the final answer.\n",
      "Final Answer: Harry Styles is Olivia Wilde's boyfriend and his current age raised to the 0.23 power is 2.169459462491557.\u001b[0m\n",
      "\n",
      "\u001b[1m> Finished chain.\u001b[0m\n"
     ]
    }
   ],
   "source": [
    "os.environ[\"LANGCHAIN_TRACING\"] = \"true\"\n",
    "\n",
    "# Both of the agent runs will be traced because the environment variable is set\n",
    "agent.run(questions[0])\n",
    "with tracing_enabled() as session:\n",
    "    assert session\n",
    "    agent.run(questions[1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "a6fd6026-dc1e-4d48-893d-3592539c7828",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "\n",
      "\u001b[1m> Entering new AgentExecutor chain...\u001b[0m\n",
      "\u001b[32;1m\u001b[1;3m I need to find out who won the US Open men's final in 2019 and then calculate his age raised to the 0.334 power.\n",
      "Action: Search\n",
      "Action Input: \"US Open men's final 2019 winner\"\u001b[0m\n",
      "Observation: \u001b[33;1m\u001b[1;3mRafael Nadal defeated Daniil Medvedev in the final, 7–5, 6–3, 5–7, 4–6, 6–4 to win the men's singles tennis title at the 2019 US Open. It was his fourth US ...\u001b[0m\n",
      "Thought:\u001b[32;1m\u001b[1;3m I need to find out the age of the winner\n",
      "Action: Search\n",
      "Action Input: \"Rafael Nadal age\"\u001b[0m\n",
      "Observation: \u001b[33;1m\u001b[1;3m36 years\u001b[0m\n",
      "Thought:\u001b[32;1m\u001b[1;3m I need to calculate the age raised to the 0.334 power\n",
      "Action: Calculator\n",
      "Action Input: 36^0.334\u001b[0m\n",
      "Observation: \u001b[36;1m\u001b[1;3mAnswer: 3.3098250249682484\u001b[0m\n",
      "Thought:\u001b[32;1m\u001b[1;3m I now know the final answer\n",
      "Final Answer: Rafael Nadal, aged 36, won the US Open men's final in 2019 and his age raised to the 0.334 power is 3.3098250249682484.\u001b[0m\n",
      "\n",
      "\u001b[1m> Finished chain.\u001b[0m\n",
      "\n",
      "\n",
      "\u001b[1m> Entering new AgentExecutor chain...\u001b[0m\n",
      "\u001b[32;1m\u001b[1;3m I need to find out who Olivia Wilde's boyfriend is and then calculate his age raised to the 0.23 power.\n",
      "Action: Search\n",
      "Action Input: \"Olivia Wilde boyfriend\"\u001b[0m\n",
      "Observation: \u001b[33;1m\u001b[1;3mSudeikis and Wilde's relationship ended in November 2020. Wilde was publicly served with court documents regarding child custody while she was presenting Don't Worry Darling at CinemaCon 2022. In January 2021, Wilde began dating singer Harry Styles after meeting during the filming of Don't Worry Darling.\u001b[0m\n",
      "Thought:\u001b[32;1m\u001b[1;3m I need to find out Harry Styles' age.\n",
      "Action: Search\n",
      "Action Input: \"Harry Styles age\"\u001b[0m\n",
      "Observation: \u001b[33;1m\u001b[1;3m29 years\u001b[0m\n",
      "Thought:\u001b[32;1m\u001b[1;3m I need to calculate 29 raised to the 0.23 power.\n",
      "Action: Calculator\n",
      "Action Input: 29^0.23\u001b[0m\n",
      "Observation: \u001b[36;1m\u001b[1;3mAnswer: 2.169459462491557\u001b[0m\n",
      "Thought:\u001b[32;1m\u001b[1;3m I now know the final answer.\n",
      "Final Answer: Harry Styles is Olivia Wilde's boyfriend and his current age raised to the 0.23 power is 2.169459462491557.\u001b[0m\n",
      "\n",
      "\u001b[1m> Finished chain.\u001b[0m\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "\"Harry Styles is Olivia Wilde's boyfriend and his current age raised to the 0.23 power is 2.169459462491557.\""
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Now, we unset the environment variable and use a context manager.\n",
    "\n",
    "if \"LANGCHAIN_TRACING\" in os.environ:\n",
    "    del os.environ[\"LANGCHAIN_TRACING\"]\n",
    "\n",
    "# here, we are writing traces to \"my_test_session\"\n",
    "with tracing_enabled(\"my_test_session\") as session:\n",
    "    assert session\n",
    "    agent.run(questions[0])  # this should be traced\n",
    "\n",
    "agent.run(questions[1])  # this should not be traced"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "9383a351-4983-44e9-abd7-ef942e1c65c4",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "\n",
      "\u001b[1m> Entering new AgentExecutor chain...\u001b[0m\n",
      "\n",
      "\u001b[1m> Entering new AgentExecutor chain...\u001b[0m\n",
      "\n",
      "\n",
      "\u001b[1m> Entering new AgentExecutor chain...\u001b[0m\n",
      "\n",
      "\u001b[32;1m\u001b[1;3m I need to find out who won the grand prix and then calculate their age raised to the 0.23 power.\n",
      "Action: Search\n",
      "Action Input: \"Formula 1 Grand Prix Winner\"\u001b[0m\u001b[32;1m\u001b[1;3m I need to find out who won the US Open men's final in 2019 and then calculate his age raised to the 0.334 power.\n",
      "Action: Search\n",
      "Action Input: \"US Open men's final 2019 winner\"\u001b[0m\u001b[33;1m\u001b[1;3mRafael Nadal defeated Daniil Medvedev in the final, 7–5, 6–3, 5–7, 4–6, 6–4 to win the men's singles tennis title at the 2019 US Open. It was his fourth US ...\u001b[0m\u001b[32;1m\u001b[1;3m I need to find out who Olivia Wilde's boyfriend is and then calculate his age raised to the 0.23 power.\n",
      "Action: Search\n",
      "Action Input: \"Olivia Wilde boyfriend\"\u001b[0m\u001b[33;1m\u001b[1;3mSudeikis and Wilde's relationship ended in November 2020. Wilde was publicly served with court documents regarding child custody while she was presenting Don't Worry Darling at CinemaCon 2022. In January 2021, Wilde began dating singer Harry Styles after meeting during the filming of Don't Worry Darling.\u001b[0m\u001b[33;1m\u001b[1;3mLewis Hamilton has won 103 Grands Prix during his career. He won 21 races with McLaren and has won 82 with Mercedes. Lewis Hamilton holds the record for the ...\u001b[0m\u001b[32;1m\u001b[1;3m I need to find out the age of the winner\n",
      "Action: Search\n",
      "Action Input: \"Rafael Nadal age\"\u001b[0m\u001b[33;1m\u001b[1;3m36 years\u001b[0m\u001b[32;1m\u001b[1;3m I need to find out Harry Styles' age.\n",
      "Action: Search\n",
      "Action Input: \"Harry Styles age\"\u001b[0m\u001b[32;1m\u001b[1;3m I need to find out Lewis Hamilton's age\n",
      "Action: Search\n",
      "Action Input: \"Lewis Hamilton Age\"\u001b[0m\u001b[33;1m\u001b[1;3m29 years\u001b[0m\u001b[32;1m\u001b[1;3m I need to calculate the age raised to the 0.334 power\n",
      "Action: Calculator\n",
      "Action Input: 36^0.334\u001b[0m\u001b[32;1m\u001b[1;3m I need to calculate 29 raised to the 0.23 power.\n",
      "Action: Calculator\n",
      "Action Input: 29^0.23\u001b[0m\u001b[36;1m\u001b[1;3mAnswer: 3.3098250249682484\u001b[0m\u001b[36;1m\u001b[1;3mAnswer: 2.169459462491557\u001b[0m\u001b[33;1m\u001b[1;3m38 years\u001b[0m\n",
      "\u001b[1m> Finished chain.\u001b[0m\n",
      "\n",
      "\u001b[1m> Finished chain.\u001b[0m\n",
      "\u001b[32;1m\u001b[1;3m I now need to calculate 38 raised to the 0.23 power\n",
      "Action: Calculator\n",
      "Action Input: 38^0.23\u001b[0m\u001b[36;1m\u001b[1;3mAnswer: 2.3086081644669734\u001b[0m\n",
      "\u001b[1m> Finished chain.\u001b[0m\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "\"Rafael Nadal, aged 36, won the US Open men's final in 2019 and his age raised to the 0.334 power is 3.3098250249682484.\""
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# The context manager is concurrency safe:\n",
    "if \"LANGCHAIN_TRACING\" in os.environ:\n",
    "    del os.environ[\"LANGCHAIN_TRACING\"]\n",
    "\n",
    "# start a background task\n",
    "task = asyncio.create_task(agent.arun(questions[0]))  # this should not be traced\n",
    "with tracing_enabled() as session:\n",
    "    assert session\n",
    "    tasks = [agent.arun(q) for q in questions[1:3]]  # these should be traced\n",
    "    await asyncio.gather(*tasks)\n",
    "\n",
    "await task"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "254fef1b-6b6e-4352-9cf4-363fba895ac7",
   "metadata": {},
   "source": [
    "## Token Counting\n",
    "LangChain offers a context manager that allows you to count tokens."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "5c3e0b89-2c5e-4036-bdf2-fb6b750e360c",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "from langchain.callbacks import get_openai_callback\n",
    "\n",
    "llm = OpenAI(temperature=0)\n",
    "with get_openai_callback() as cb:\n",
    "    llm(\"What is the square root of 4?\")\n",
    "\n",
    "total_tokens = cb.total_tokens\n",
    "assert total_tokens > 0\n",
    "\n",
    "with get_openai_callback() as cb:\n",
    "    llm(\"What is the square root of 4?\")\n",
    "    llm(\"What is the square root of 4?\")\n",
    "\n",
    "assert cb.total_tokens == total_tokens * 2\n",
    "\n",
    "# You can kick off concurrent runs from within the context manager\n",
    "with get_openai_callback() as cb:\n",
    "    await asyncio.gather(\n",
    "        *[llm.agenerate([\"What is the square root of 4?\"]) for _ in range(3)]\n",
    "    )\n",
    "\n",
    "assert cb.total_tokens == total_tokens * 3\n",
    "\n",
    "# The context manager is concurrency safe\n",
    "task = asyncio.create_task(llm.agenerate([\"What is the square root of 4?\"]))\n",
    "with get_openai_callback() as cb:\n",
    "    await llm.agenerate([\"What is the square root of 4?\"])\n",
    "\n",
    "await task\n",
    "assert cb.total_tokens == total_tokens"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
