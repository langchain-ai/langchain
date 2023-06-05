{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "12f2b84c",
   "metadata": {},
   "source": [
    "# Getting Started\n",
    "\n",
    "One of the core value props of LangChain is that it provides a standard interface to models. This allows you to swap easily between models. At a high level, there are two main types of models: \n",
    "\n",
    "- Language Models: good for text generation\n",
    "- Text Embedding Models: good for turning text into a numerical representation\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a5d0965c",
   "metadata": {},
   "source": [
    "## Language Models\n",
    "\n",
    "There are two different sub-types of Language Models: \n",
    "    \n",
    "- LLMs: these wrap APIs which take text in and return text\n",
    "- ChatModels: these wrap models which take chat messages in and return a chat message\n",
    "\n",
    "This is a subtle difference, but a value prop of LangChain is that we provide a unified interface accross these. This is nice because although the underlying APIs are actually quite different, you often want to use them interchangeably.\n",
    "\n",
    "To see this, let's look at OpenAI (a wrapper around OpenAI's LLM) vs ChatOpenAI (a wrapper around OpenAI's ChatModel)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "3c932182",
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain.llms import OpenAI\n",
    "from langchain.chat_models import ChatOpenAI"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "b90db85d",
   "metadata": {},
   "outputs": [],
   "source": [
    "llm = OpenAI()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "61ef89e4",
   "metadata": {},
   "outputs": [],
   "source": [
    "chat_model = ChatOpenAI()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fa14db90",
   "metadata": {},
   "source": [
    "### `text` -> `text` interface"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "2d9f9f89",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'\\n\\nHi there!'"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "llm.predict(\"say hi!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "4dbef65b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'Hello there!'"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "chat_model.predict(\"say hi!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b67ea8a1",
   "metadata": {},
   "source": [
    "### `messages` -> `message` interface"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "066dad10",
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain.schema import HumanMessage"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "67b95fa5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "AIMessage(content='\\n\\nHello! Nice to meet you!', additional_kwargs={}, example=False)"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "llm.predict_messages([HumanMessage(content=\"say hi!\")])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "f5ce27db",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "AIMessage(content='Hello! How can I assist you today?', additional_kwargs={}, example=False)"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "chat_model.predict_messages([HumanMessage(content=\"say hi!\")])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3457a70e",
   "metadata": {},
   "outputs": [],
   "source": []
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
   "version": "3.9.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
