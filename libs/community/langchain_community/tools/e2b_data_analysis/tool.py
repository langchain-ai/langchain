from __future__ import annotations

import base64
import os
from typing import Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import Tool
from e2b_code_interpreter import Sandbox
from IPython.display import display, Image


class E2BDataAnalyzer:
    """LLM-powered tool for analyzing CSV data using E2B sandbox and GPT-4."""
    
    def __init__(
        self,
        e2b_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        timeout: int = 60,
        model: str = "gpt-4",
        temperature: float = 0.5
    ):
        """
        Initialize the analyzer with API keys and configuration.
        
        Args:
            e2b_api_key: API key for E2B sandbox
            openai_api_key: API key for OpenAI
            timeout: Sandbox timeout in seconds
            model: OpenAI model to use (default: gpt-4)
            temperature: LLM temperature setting (default: 0.5)
        """
        self.sandbox = Sandbox(api_key=e2b_api_key)
        self.sandbox.set_timeout(timeout)
        
        # Initialize the LLM
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model,
            temperature=temperature
        )
        
        # Create the tool function
        def run_analysis(code: str) -> dict:
            """Execute Python code in the sandbox and handle results."""
            execution = self.sandbox.run_code(code)
            
            if execution.error:
                return {
                    "error": execution.error.name,
                    "value": execution.error.value,
                    "traceback": execution.error.traceback
                }

            results = []
            for idx, result in enumerate(execution.results):
                if result.png:
                    file_name = f"chart-{idx}.png"
                    with open(file_name, 'wb') as f:
                        f.write(base64.b64decode(result.png))
                    display(Image(filename=file_name))
                    results.append(file_name)

            return {"images": results} if results else {"message": "Analysis completed successfully"}

        # Create the tool using the Tool class directly
        self.tools = [
            Tool(
                name="run_analysis",
                func=run_analysis,
                description="Execute Python code in the sandbox and handle results"
            )
        ]
# Define the prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a data analysis expert who converts natural language queries into Python code. 
            You use pandas for data manipulation and matplotlib for visualization. 
            Always create clear, informative visualizations with proper titles, labels, and legends when appropriate.
            Use seaborn styling for better-looking visualizations.
            The data is already loaded into a pandas DataFrame called 'df'."""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        # Create the agent
        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt_template)
        self.executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

    
    def analyze(self, query: str, csv_path: str) -> dict:
        """
        Analyze CSV data based on a natural language query.
        
        Args:
            query: Natural language question or analysis request
            csv_path: Path to the local CSV file
            
        Returns:
            dict: Results of the analysis including any generated images
        """
        # Upload the CSV to the sandbox
        with open(csv_path, "rb") as f:
            dataset_path = self.sandbox.files.write("dataset.csv", f)
        
        # Load the CSV locally to get schema information
        import pandas as pd
        data = pd.read_csv(csv_path)
        columns = ", ".join(data.columns)
        sample_rows = data.head(3).to_dict(orient="records")
        
       # Create the analysis prompt
        prompt_input = {
            "input": f"The user asks: '{query}'\n\nThe dataset is located at {dataset_path.path}\nColumns available: {columns}\nSample data: {sample_rows}\n\nGenerate and execute Python code to answer this question. Always:\n1. Start by reading the CSV: df = pd.read_csv('{dataset_path.path}')\n2. Include appropriate data cleaning if needed\n3. Create informative visualizations with proper labels and seaborn styling\n4. Print relevant statistics or insights\n5. Use plt.figure(figsize=(12, 6)) for better sized plots\n6. Add plt.tight_layout() before plt.show()"
        }
        
        # Run the analysis
        response = self.executor.invoke(prompt_input)
        return response
    def close(self):
        """Clean up resources."""
        self.sandbox.close()


