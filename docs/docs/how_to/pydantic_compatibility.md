# How to use LangChain with different Pydantic versions

As of the `0.3` release, LangChain uses Pydantic 2 internally. 

Users should install Pydantic 2 and are advised to **avoid** using the `pydantic.v1` namespace of Pydantic 2 with
LangChain APIs.

If you're working with prior versions of LangChain, please see the following guide
on [Pydantic compatibility](https://python.langchain.com/v0.2/docs/how_to/pydantic_compatibility).