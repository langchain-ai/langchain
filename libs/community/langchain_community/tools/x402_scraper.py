import json
import base64
import time
import requests
from typing import Optional, Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class ScraperInput(BaseModel):
    url: str = Field(description="The URL to scrape into markdown")

class X402MarkdownScraperTool(BaseTool):
    name: str = "x402_markdown_scraper"
    description: str = "Fetches a URL and returns clean markdown. Inherently protected from rate limits via x402 machine-to-machine payment."
    args_schema: Type[BaseModel] = ScraperInput
    
    def _run(self, url: str) -> str:
        target_url = f"https://x402-api-middleware.onrender.com/scrape?url={url}"
        
        resp = requests.get(target_url)
        if resp.status_code != 402:
            return resp.text
            
        try:
            from web3 import Web3
            from eth_account import Account
            from eth_account.messages import encode_typed_data
        except ImportError:
            return "Error: Please install web3 and eth_account to use x402 payment resolution."
            
        account = Account.create()
        price = 0.05
        nonce = int(time.time() * 1000)
        domain_data = {
            "name": "x402",
            "version": "2",
            "chainId": 8453,
            "verifyingContract": "0xDc64a140Aa3E981100a9becA4E685f962f0cF6C9",
        }
        message_types = {
            "Voucher": [
                {"name": "agentAddress", "type": "address"},
                {"name": "amountUsdc", "type": "uint256"},
                {"name": "nonce", "type": "uint256"}
            ]
        }
        message_data = {
            "agentAddress": account.address,
            "amountUsdc": int(price * 1e6),
            "nonce": nonce
        }
        
        signable_message = encode_typed_data(domain_data, message_types, message_data)
        signature = account.sign_message(signable_message).signature.hex()
        
        legacy_payload = {
            "agentAddress": account.address,
            "amountUsdc": price,
            "nonce": nonce,
            "signature": signature
        }
        sig_header = base64.b64encode(json.dumps(legacy_payload).encode()).decode()
        
        final_resp = requests.get(target_url, headers={"PAYMENT-SIGNATURE": sig_header})
        return final_resp.text
