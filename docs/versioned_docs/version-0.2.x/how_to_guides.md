# "How-to" guides

Here you’ll find short answers to “How do I….?” types of questions. 
These how-to guides don’t cover topics in depth – you’ll find that material in the [Tutorials](/docs/tutorials) and the [API Reference](https://api.python.langchain.com/en/latest/). 
However, these guides will help you quickly accomplish common tasks.

## Core Functionality

- How to return structured data from an LLM
- How to use an LLM to call tools
- [How to stream](/docs/how_to/streaming)
- How to see what is going on inside your LLM application
- How to test your LLM application
- How to deploy your LLM application

### Tool Usage

- [How to use tools in a chain](/docs/how_to/tool_chain)
- [How to use agents to use tools](/docs/how_to/tools_agents)
- [How to use tools without function calling](/docs/how_to/tools_prompting)
- [How to let the LLM choose between multiple tools](/docs/how_to/tools_multiple)
- [How to add a human in the loop to tool usage](/docs/how_to/tools_human)
- [How to do parallel tool use](/docs/how_to/tools_parallel)
- [How to handle errors when calling tools](/docs/how_to/tools_error)

## LangChain Expression Language (LCEL)

- [How to chain runnables](/docs/how_to/sequence)
- [How to run two runnables in parallel](/docs/how_to/parallel/)
- [How to attach runtime arguments to a runnable](/docs/how_to/binding/)
- [How to run custom functions](/docs/how_to/functions)
- [How to pass through arguments from one step to the next](/docs/how_to/passthrough)
- [How to add values to state](/docs/how_to/assign)
- [How to configure runtime chain internals](/docs/how_to/configure)
- [How to add message history](/docs/how_to/message_history)
- [How to do routing](/docs/how_to/routing)
- [How to inspect your runnables](/docs/how_to/inspect)
- [How to use `@chain` decorator to create a runnable](/docs/how_to/decorator)
- [How to manage prompt size](/docs/how_to/prompt_size)
- [How to string together multiple chains](/docs/how_to/multiple_chains)

## Components

### Prompts
- [How to use few shot examples](/docs/how_to/few_shot_examples)
- [How to use few shot examples in chat models](/docs/how_to/few_shot_examples_chat/)
- [How to partial prompt templates](/docs/how_to/prompts_partial)
- [How to compose two prompts together](/docs/how_to/prompts_composition)

### Example Selectors
- [How to use example selectors](/docs/how_to/example_selectors.ipynb)
- [How to select examples by length](/docs/how_to/example_selectors_length_based.ipynb)
- [How to select examples by semantic similarity](/docs/how_to/example_selectors_similarity.ipynb)
- [How to select examples by semantic ngram overlap](/docs/how_to/example_selectors_ngram.ipynb)
- [How to select examples by maximal marginal relevance](/docs/how_to/example_selectors_mmr.ipynb)

### Chat Models
- [How to function/tool calling](/docs/modules/model_io/chat/function_calling)
- [How to get models to return structured output](/docs/modules/model_io/chat/structured_output)
- [How to cache model responses](/docs/modules/model_io/chat/chat_model_caching)
- [How to get log probabilities](/docs/modules/model_io/chat/logprobs)
- [How to create a custom chat model class](/docs/modules/model_io/chat/custom_chat_model)
- [How to stream a response back](/docs/modules/model_io/chat/streaming)
- [How to track token usage](/docs/modules/model_io/chat/token_usage_tracking)

### LLMs
- [How to cache model responses](/docs/modules/model_io/llms/llm_caching)
- [How to create a custom LLM class](/docs/modules/model_io/llms/custom_llm)
- [How to stream a response back](/docs/modules/model_io/llms/streaming_llm)
- [How to track token usage](/docs/modules/model_io/llms/token_usage_tracking)

### Output Parsers
- [How to use output parsers to parse an LLM response into structured format](/docs/modules/model_io/output_parsers/quick_start)
- [How to parse JSON output](/docs/modules/model_io/output_parsers/types/json)
- [How to parse XML output](/docs/modules/model_io/output_parsers/types/xml)
- [How to parse YAML output](/docs/modules/model_io/output_parsers/types/yaml)
- [How to retry when output parsing errors occur](/docs/modules/model_io/output_parsers/types/retry)
- [How to try to fix errors in output parsing](/docs/modules/model_io/output_parsers/types/output_fixing)
- [How to write a custom output parser class](/docs/modules/model_io/output_parsers/custom)

### Document Loaders
- [How to load CSV data](/docs/modules/data_connection/document_loaders/csv)
- [How to load data from a directory](/docs/modules/data_connection/document_loaders/file_directory)
- [How to load HTML data](/docs/modules/data_connection/document_loaders/html)
- [How to load JSON data](/docs/modules/data_connection/document_loaders/json)
- [How to load Markdown data](/docs/modules/data_connection/document_loaders/markdown)
- [How to load Microsoft Office data](/docs/modules/data_connection/document_loaders/office_file)
- [How to load PDF files](/docs/modules/data_connection/document_loaders/pdf)
- [How to write a custom document loader](/docs/modules/data_connection/document_loaders/custom)

### Text Splitter
- [How to recursively split text](/docs/modules/data_connection/document_transformers/recursive_text_splitter)
- [How to split by HTML headers](/docs/modules/data_connection/document_transformers/HTML_header_metadata)
- [How to split by HTML sections](/docs/modules/data_connection/document_transformers/HTML_section_aware_splitter)
- [How to split by character](/docs/modules/data_connection/document_transformers/character_text_splitter)
- [How to split code](/docs/modules/data_connection/document_transformers/code_splitter)
- [How to split Markdown by headers](/docs/modules/data_connection/document_transformers/markdown_header_metadata)
- [How to recursively split JSON](/docs/modules/data_connection/document_transformers/recursive_json_splitter)
- [How to split text into semantic chunks](/docs/modules/data_connection/document_transformers/semantic-chunker)
- [How to split by tokens](/docs/modules/data_connection/document_transformers/split_by_token)

### Embedding Models
- [How to embed text data](/docs/modules/data_connection/text_embedding)
- [How to cache embedding results](/docs/modules/data_connection/text_embedding/caching_embeddings)

### Vector Stores
- [How to use a vector store to retrieve data](/docs/modules/data_connection/vectorstores)

### Retrievers
- [How use a vector store to retrieve data](/docs/modules/data_connection/retrievers/vectorstore)
- [How to generate multiple queries to retrieve data for](/docs/modules/data_connection/retrievers/MultiQueryRetriever)
- [How to use contextual compression to compress the data retrieved](/docs/modules/data_connection/retrievers/contextual_compression)
- [How to write a custom retriever class](/docs/modules/data_connection/retrievers/custom_retriever)
- [How to combine the results from multiple retrievers](/docs/modules/data_connection/retrievers/ensemble)
- [How to reorder retrieved results to put most relevant documents not in the middle](/docs/modules/data_connection/retrievers/long_context_reorder)
- [How to generate multiple embeddings per document](/docs/modules/data_connection/retrievers/multi_vector)
- [How to retrieve the whole document for a chunk](/docs/modules/data_connection/retrievers/parent_document_retriever)
- [How to generate metadata filters](/docs/modules/data_connection/retrievers/self_query)
- [How to create a time-weighted retriever](/docs/modules/data_connection/retrievers/time_weighted_vectorstore)

### Indexing
- [How to reindex data to keep your vectorstore in-sync with the underlying data source](/docs/modules/data_connection/indexing)

### Tools
- [How to use LangChain tools](/docs/modules/tools)
- [How to use LangChain toolkits](/docs/modules/tools/toolkits)
- [How to define a custom tool](/docs/modules/tools/custom_tools)
- [How to convert LangChain tools to OpenAI functions](/docs/modules/tools/tools_as_openai_functions)

### Agents
- [How to create a custom agent](/docs/modules/agents/how_to/custom_agent)
- [How to stream responses from an agent](/docs/modules/agents/how_to/streaming)
- [How to run an agent as an iterator](/docs/modules/agents/how_to/agent_iter)
- [How to return structured output from an agent](/docs/modules/agents/how_to/agent_structured)
- [How to handle parsing errors in an agent](/docs/modules/agents/how_to/handle_parsing_errors)
- [How to access intermediate steps](/docs/modules/agents/how_to/intermediate_steps)
- [How to cap the maximum number of iterations](/docs/modules/agents/how_to/max_iterations)
- [How to set a time limit for agents](/docs/modules/agents/how_to/max_time_limit)

## Use Cases

### Q&A with RAG
- [How to add chat history](/docs/use_cases/question_answering/chat_history/)
- [How to stream](/docs/use_cases/question_answering/streaming/)
- [How to return sources](/docs/use_cases/question_answering/sources/)
- [How to return citations](/docs/use_cases/question_answering/citations/)
- [How to do per-user retrieval](/docs/use_cases/question_answering/per_user/)


### Extraction
- [How to use reference examples](/docs/use_cases/extraction/how_to/examples/)
- [How to handle long text](/docs/use_cases/extraction/how_to/handle_long_text/)
- [How to do extraction without using function calling](/docs/use_cases/extraction/how_to/parse)

### Chatbots
- [How to manage memory](/docs/use_cases/chatbots/memory_management)
- [How to do retrieval](/docs/use_cases/chatbots/retrieval)
- [How to use tools](/docs/use_cases/chatbots/tool_usage)

### Query Analysis
- [How to add examples to the prompt](/docs/use_cases/query_analysis/how_to/few_shot)
- [How to handle cases where no queries are generated](/docs/use_cases/query_analysis/how_to/no_queries)
- [How to handle multiple queries](/docs/use_cases/query_analysis/how_to/multiple_queries)
- [How to handle multiple retrievers](/docs/use_cases/query_analysis/how_to/multiple_retrievers)
- [How to construct filters](/docs/use_cases/query_analysis/how_to/constructing-filters)
- [How to deal with high cardinality categorical variables](/docs/use_cases/query_analysis/how_to/high_cardinality)

### Q&A over SQL + CSV
- [How to use prompting to improve results](/docs/use_cases/sql/prompting)
- [How to do query validation](/docs/use_cases/sql/query_checking)
- [How to deal with large databases](/docs/use_cases/sql/large_db)
- [How to deal with CSV files](/docs/use_cases/sql/csv)

### Q&A over Graph Databases
- [How to map values to a database](/docs/use_cases/graph/mapping)
- [How to add a semantic layer over the database](/docs/use_cases/graph/semantic)
- [How to improve results with prompting](/docs/use_cases/graph/prompting)
- [How to construct knowledge graphs](/docs/use_cases/graph/constructing)
