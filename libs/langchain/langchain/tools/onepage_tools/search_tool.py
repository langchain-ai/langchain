# onepage_agent/tools/search_tool.py
import requests
from langchain.tools import BaseTool

# Tool for company search
class SearchTool(BaseTool):
    name: str = "Search"
    description: str = "Search for a company or person. Provide inputs as a string in the format 'Search for <query> using token <token>', or as a dictionary with 'token' and 'input' keys."

    def _run(self, inputs: dict):

        """
        Executes a search query using the provided token.

        :param search: The search query (name, email, LinkedIn, domain).
        :param
        """
        

        if isinstance(inputs, dict):
            token = inputs.get("token")
            search = inputs.get("input")
        elif isinstance(inputs, str):
            # Parse inputs from a string if provided in natural language
            try:
                parts = inputs.split(" using token ")
                search = parts[0].replace("Search for ", "").strip()
                token = parts[1].strip() if len(parts) > 1 else None
            except Exception:
                return "Error: Unable to parse inputs. Provide in format 'Search for <query> using token <token>'."
        else:
            return "Error: Inputs must be a dictionary or properly formatted string."

        if not token or not search:
            return "Error: Missing required inputs 'token' or 'input'."

        url = f"<base_url>/F6DBDEC5"
        search = search.strip("'")
        token = token.strip("'")
        payload = {"s": search, "l": "en", "o": "content"}
        headers = {
            "Content-Type": "application/json",
            "authToken": token,  # Replace with actual token management
        }

        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 200:
            try:
                data = response.json()
                if data.get("success"):
                    if data.get("type") == "no_credits":
                        return f"\nYou have used all credits. Purchase from <api_key_url>"
                    if data.get("type") == "error":
                        return f"\nSorry couldn't find the results. Try again with same or different input"
                else:
                    return f"Search failed: {data.get('message', 'No error message provided')}"
            except ValueError:
                return f"\n{response.text}"
        else:
            return f"Error during search: {response.text}"

    def _arun(self, search: str):
        raise NotImplementedError("Async not implemented for SearchTool.")
