# langchain-0g

This package contains the LangChain integrations for 0G.ai through their `a0g` SDK.

## Installation and Setup

* Install the 0G.ai Python package:

```bash
pip install python-0g langchain_0g
```

* Set your 0G.ai wallet private key as an environment variable:

```bash
export A0G_PRIVATE_KEY="your_wallet_private_key"
```

## Chat model

`ZGChat` is a wrapper around 0G.ai chat models. It allows you to interact with 0G.ai-hosted LLMs in a LangChain-compatible way.

```python
from langchain_0g import ZGChat

chat = ZGChat(provider="0xf07240Efa67755B5311bc75784a061eDB47165Dd")
response = chat.invoke("Hello 0G!")
print(response)
```

* `provider` is the ENS address of the 0G.ai model.
* `private_key` is optional if you have `A0G_PRIVATE_KEY` in the environment.

### Asynchronous Usage

```python
import asyncio
from langchain_0g import ZGChat

async def main():
    chat = ZGChat(provider="0xf07240Efa67755B5311bc75784a061eDB47165Dd")
    response = await chat.ainvoke("Hi!")
    print(response)

asyncio.run(main())
```

---

## LLM (Legacy)

`ZGLLM` refers to legacy text-completion models hosted on 0G.ai.

```python
from langchain_0g import ZGLLM

llm = ZGLLM(provider="0xf07240Efa67755B5311bc75784a061eDB47165Dd")
text = llm.invoke("Summarize the following article...")
print(text)
```

* `provider` specifies the 0G.ai model.
* Use `private_key` to sign requests via your wallet.

---

## List available models

```python
from langchain_0g import ZGChat

llm = ZGChat(provider="0xf07240Efa67755B5311bc75784a061eDB47165Dd")
print(llm.zg_client.get_all_services())

```

---

## Notes

* 0G.ai clients (`client` and `async_client`) handle authentication using your wallet private key instead of traditional OpenAI API keys.
* The ENS `provider` identifies which LLM to query.
* All requests go through 0G.ai smart contracts for on-chain verification.
