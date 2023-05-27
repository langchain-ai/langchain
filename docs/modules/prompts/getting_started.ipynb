{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "3651e424",
   "metadata": {},
   "source": [
    "# Getting Started\n",
    "\n",
    "This section contains everything related to prompts. A prompt is the value passed into the Language Model. This value can either be a string (for LLMs) or a list of messages (for Chat Models).\n",
    "\n",
    "The data types of these prompts are rather simple, but their construction is anything but. Value props of LangChain here include:\n",
    "\n",
    "- A standard interface for string prompts and message prompts\n",
    "- A standard (to get started) interface for string prompt templates and message prompt templates\n",
    "- Example Selectors: methods for inserting examples into the prompt for the language model to follow\n",
    "- OutputParsers: methods for inserting instructions into the prompt as the format in which the language model should output information, as well as methods for then parsing that string output into a format.\n",
    "\n",
    "We have in depth documentation for specific types of string prompts, specific types of chat prompts, example selectors, and output parsers.\n",
    "\n",
    "Here, we cover a quick-start for a standard interface for getting started with simple prompts."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ff34414d",
   "metadata": {},
   "source": [
    "## PromptTemplates\n",
    "\n",
    "PromptTemplates are responsible for constructing a prompt value. These PromptTemplates can do things like formatting, example selection, and more. At a high level, these are basically objects that expose a `format_prompt` method for constructing a prompt. Under the hood, ANYTHING can happen."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "7ce42639",
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain.prompts import PromptTemplate, ChatPromptTemplate"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "5a178697",
   "metadata": {},
   "outputs": [],
   "source": [
    "string_prompt = PromptTemplate.from_template(\"tell me a joke about {subject}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "f4ef6d6b",
   "metadata": {},
   "outputs": [],
   "source": [
    "chat_prompt = ChatPromptTemplate.from_template(\"tell me a joke about {subject}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "5f16c8f1",
   "metadata": {},
   "outputs": [],
   "source": [
    "string_prompt_value = string_prompt.format_prompt(subject=\"soccer\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "863755ea",
   "metadata": {},
   "outputs": [],
   "source": [
    "chat_prompt_value = chat_prompt.format_prompt(subject=\"soccer\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8b3d8511",
   "metadata": {},
   "source": [
    "## `to_string`\n",
    "\n",
    "This is what is called when passing to an LLM (which expects raw text)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "1964a8a0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'tell me a joke about soccer'"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "string_prompt_value.to_string()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "bf6c94e9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'Human: tell me a joke about soccer'"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "chat_prompt_value.to_string()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c0825af8",
   "metadata": {},
   "source": [
    "## `to_messages`\n",
    "\n",
    "This is what is called when passing to ChatModel (which expects a list of messages)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "e4da46f3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[HumanMessage(content='tell me a joke about soccer', additional_kwargs={}, example=False)]"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "string_prompt_value.to_messages()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "eae84b88",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[HumanMessage(content='tell me a joke about soccer', additional_kwargs={}, example=False)]"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "chat_prompt_value.to_messages()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a34fa440",
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
