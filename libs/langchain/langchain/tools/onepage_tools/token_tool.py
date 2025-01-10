# onepage_agent/tools/token_tool.py
import requests
from langchain.tools import BaseTool

# Tool for token validation
class TokenValidationTool(BaseTool):
    name: str = "TokenValidation"
    description: str = "Validates the user's token and retrieves plan details."

    def _run(self, query: str):
        url = f"<base_url>/0EA1B4B0"
        payload = {"token": query}
        headers = {"Content-Type": "application/json"}

        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                response_type = data.get("type")
                if response_type == "credits_data":
                    credits = data.get("data", {}).get("Credits", "N/A")
                    return f"Token validated successfully. Credits available: {credits}"
                elif response_type == "no_credits":
                    return "Token is valid but no credits are available. Please buy credits to proceed."
                elif response_type == "invalid":
                    return "Invalid token. Please try again or exit."
                else:
                    return "Unexpected response type."
            else:
                return f"Token validation failed: {data.get('message')}"
        else:
            return f"Error validating token: {response.text}"

    def _arun(self, query: str):
        raise NotImplementedError("Async not implemented for TokenValidationTool.")
