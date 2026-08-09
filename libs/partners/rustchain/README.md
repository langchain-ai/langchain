# langchain-rustchain

Integration of [RustChain](https://rustchain.org) — a DePIN
Proof-of-Antiquity blockchain whose HTTP API is agent-native (no auth, no
captcha, wallet = any string) — as a native LangChain tool.

> Based on the open bounty:
> <https://github.com/Scottcjn/rustchain-bounties/issues/3074>

## Installation

```bash
pip install -U langchain-rustchain
```

The public node requires no API key. Optional environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `RUSTCHAIN_HOST` | `https://rustchain.org` | Node base URL |
| `RUSTCHAIN_SSL_VERIFY` | `1` | `0` to skip TLS verification (self-signed node) |
| `GITHUB_TOKEN` | *(none)* | Raises GitHub API rate limit for `list_bounties` |

## Usage

```python
from langchain_rustchain import RustChainTool

tool = RustChainTool()

balance = tool.check_balance("demo")          # -> float (RTC)
bounties = tool.list_bounties(limit=10)        # -> list[dict]
health = tool.get_node_health()                # -> dict
epoch = tool.get_current_epoch()               # -> dict

# or through the BaseTool interface (JSON command string):
print(tool._run('{"method": "get_node_health"}'))
```

## Bounty / payout note

This integration targets the bounty
[#3074](https://github.com/Scottcjn/rustchain-bounties/issues/3074)
(25 RTC). Payout wallet: `el-auto-negocio`.

## Limitations and Considerations

### GitHub API Rate Limits
The `list_bounties()` method uses the public GitHub REST API without
authentication (60 requests/hour per IP). For intensive usage, set the
`GITHUB_TOKEN` environment variable with a Personal Access Token to increase
the limit to 5000 req/h.

Example:
```bash
export GITHUB_TOKEN="ghp_your_token_here"
python -c "from langchain_rustchain import RustChainTool; \
  print(RustChainTool().list_bounties(10))"
```

### list_bounties() Implementation Note
RustChain does NOT expose an HTTP endpoint `/api/bounties` (returns 404).
Bounties exist as GitHub issues with label `bounty` in the official repo
`Scottcjn/rustchain-bounties`. The `list_bounties()` method queries the GitHub
REST API searching for those issues.

### TLS Certificate
The default RustChain node uses a self-signed certificate. For production
use, set `RUSTCHAIN_SSL_VERIFY=0` or provide a valid certificate.

## Tests

```bash
make test          # unit tests (no network)
```