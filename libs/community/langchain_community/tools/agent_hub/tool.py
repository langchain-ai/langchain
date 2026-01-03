"""LangChain integration for Hour.IT Agent Hub"""
from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
import requests


class AgentHubInput(BaseModel):
    """Input for Agent Hub service."""
    service: str = Field(description="Service name: sentiment, translate, summarize, etc.")
    params: dict = Field(description="Service parameters as a dictionary")
    payment_tx: Optional[str] = Field(default=None, description="USDC payment transaction hash on Base Mainnet")


class AgentHubTool(BaseTool):
    """Tool for calling Hour.IT Agent Hub paid AI services."""

    name = "agent_hub"
    description = """Useful for accessing premium AI services with crypto payment.
    Available services: sentiment analysis, translation, summarization, web scraping, 
    data extraction, research, content generation, code review, SEO optimization, 
    SWOT analysis, competitive analysis, and more.
    Requires USDC payment on Base Mainnet."""

    args_schema: Type[BaseModel] = AgentHubInput
    api_url: str = "https://web-production-4833.up.railway.app"

    def _run(self, service: str, params: dict, payment_tx: Optional[str] = None) -> str:
        """Execute the service call."""
        endpoint = f"{self.api_url}/agent/{service}"
        headers = {}
        if payment_tx:
            headers["PAYMENT-SIGNATURE"] = payment_tx

        response = requests.post(endpoint, json=params, headers=headers)

        if response.status_code == 402:
            payment_info = response.json()
            return f"Payment required: {payment_info}"

        return response.json()

    async def _arun(self, service: str, params: dict, payment_tx: Optional[str] = None) -> str:
        """Async version."""
        raise NotImplementedError("Use synchronous version")


# Example usage
if __name__ == "__main__":
    tool = AgentHubTool()

    # Example 1: Check pricing first
    result = tool._run("sentiment", {"text": "AI agents are amazing!"})
    print(result)
