# langchain-sarvam

Integration package connecting Sarvam AI chat completions with LangChain.

## Installation (dev, from monorepo)

```powershell
# From repo root
python -m pip install -e libs/partners/sarvam
```

Or with `uv` inside the package:

```powershell
# From libs/partners/sarvam/
uv sync
```

## Setup

```powershell
$Env:SARVAM_API_KEY = "your_api_key"
```

## Usage

```python
from langchain_sarvam import ChatSarvam

llm = ChatSarvam(model="sarvam-m", temperature=0.2, max_tokens=128)
resp = llm.invoke([("system", "You are helpful"), ("human", "Hello!")])
print(resp.content)
```

Streaming:

```python
for chunk in ChatSarvam(model="sarvam-m", streaming=True).stream("Tell me a joke"):
    print(chunk.text, end="")
```

## Testing

- On Windows, running tests with `pytest-socket` and `--disable-socket` can block asyncio event loop creation. The async test `test_sarvam_ainvoke` is marked with `@pytest.mark.enable_socket` to allow loop setup only for that test; all other tests remain socket-restricted.
- Run the suite with sockets disabled (safe for the rest of the tests):

```powershell
uv run --group test pytest --disable-socket --allow-unix-socket tests/unit_tests -q
```

Or without `uv`:

```powershell
pytest --disable-socket --allow-unix-socket tests/unit_tests -q
```

## Development

- Tests: `pytest tests/unit_tests -q`
- Type check: `mypy .`
- Lint: `ruff check .`
