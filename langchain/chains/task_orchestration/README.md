# Task Orchestration Chains

These chains are meant to aid in and implement the paradigm introduced by Yohei Nakajima in the paper [Task-driven Autonomous Agent Utilizing GPT-4, Pinecone, and LangChain for Diverse Applications](https://yoheinakajima.com/task-driven-autonomous-agent-utilizing-gpt-4-pinecone-and-langchain-for-diverse-applications/)

These chains were mostly made from the [LangChain Wiki](https://python.langchain.com/en/latest/use_cases/agents/baby_agi.html) post on the BabyAGI usecase, with some re-factoring and modifications so that the BabyAGI chain can be called in a simple manner. 

![Task Orchestration](httpswith://pbs.twimg.com/media/FsW_xF4aEAASyZH?format=png&name=small)

## BabyAGI 

The BabyAGI chain takes in two key word arguments:

`objective` (mandatory) - The overarching objective you want the task orchestration system to converge to
`first_task` (optional) - The prompt it gets for its "first task", which is usually some form of creating a task list. The default is "Make a todo list".

The `from_llm` method that constructs the chain takes in the following arguments that may be of interest:

`llm` - The LLM model you want to the chain to use. Note: Using a model like GPT-4 add up costs extremely quickly. Use with caution.
`vectorstore` - The vectorstore you want the chain to use
`max_iterations` - The maximum number of iterations, i.e. number of tasks that BabyAGI will output a result for and iterate on. If this number is not provided, the chain WILL run forever. 