# SQL with LLaMA2

This template allows you to chat with a SQL database in natural language using LLaMA2.

It is configured to use [Replicate](https://python.langchain.com/docs/integrations/llms/replicate).

But, it can be adapted to any API that support LLaMA2, including [Fireworks](https://python.langchain.com/docs/integrations/chat/fireworks) and others.

See related templates `sql-ollama` and `sql-llamacpp` for private, local chat with SQL.

## Set up SQL DB

This template includes an example DB of 2023 NBA rosters.

You can see instructions to build this DB [here](https://github.com/facebookresearch/llama-recipes/blob/main/demo_apps/StructuredLlama.ipynb).

##  LLM

This template will use a `Replicate` [hosted version](https://replicate.com/meta/llama-2-13b-chat/versions/f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d) of LLaMA2. 

Be sure that `REPLICATE_API_TOKEN` is set in your environment.