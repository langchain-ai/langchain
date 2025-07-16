* Default to asynchronous methods (ainvoke, abatch, astream) where applicable.
* Prioritize LangChain Expression Language (LCEL) for creating chains. Use the pipe operator (|) to connect components.
* Generate modular imports from specific packages like langchain_core, langchain_community, and langchain_openai. Do not import from the top-level langchain package.
* Format all generated Python code to be compliant with ruff rules.
* All generated Python code must include type hints.
* When suggesting package installation commands, use uv pip install as this project uses uv.
* When creating tools for agents, use the @tool decorator from langchain_core.tools. The tool's docstring serves as its functional description for the agent.
* Avoid suggesting deprecated components, such as the legacy LLMChain.
