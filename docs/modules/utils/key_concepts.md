# Key Concepts

## Python REPL
Sometimes, for complex calculations, rather than have an LLM generate the answer directly, 
it can be better to have the LLM generate code to calculate the answer, and then run that code to get the answer. 
In order to easily do that, we provide a simple Python REPL to execute commands in.
This interface will only return things that are printed - 
therefore, if you want to use it to calculate an answer, make sure to have it print out the answer.

## Bash
It can often be useful to have an LLM generate bash commands, and then run them. 
A common use case this is for letting it interact with your local file system. 
We provide an easy component to execute bash commands.

## Requests Wrapper
The web contains a lot of information that LLMs do not have access to. 
In order to easily let LLMs interact with that information, 
we provide a wrapper around the Python Requests module that takes in a URL and fetches data from that URL.

## Google Search
This uses the official Google Search API to look up information on the web.

## SerpAPI
This uses SerpAPI, a third party search API engine, to interact with Google Search.

## Searx Search
This uses the Searx (SearxNG fork) meta search engine API to lookup information
on the web.  It supports 139 search engines and is easy to self-host
which makes it a good choice for privacy-conscious users.
