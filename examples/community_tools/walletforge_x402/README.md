# WalletForge x402 community tools

Paid `fetch_markdown` (0.05 USDC on Base) and `normalize_text` (0.01 USDC on Base)
for LangChain agents, via [x402](https://www.x402.org/) V2.

This directory is a **cookbook-style contribution**. New third-party tools are no
longer accepted in-tree in this monorepo, and [`langchain-community` is
sunset](https://github.com/langchain-ai/langchain-community/issues/674). The
implementation lives in a standalone package, the same pattern as Coinbase
AgentKit:

- Package: https://github.com/mig26-design/walletforge-x402
- Live API: https://api.walletforge.app
- Settle proof: https://basescan.org/tx/0x5cebbe810ca7208bc85ab0231c59c03dfee967ad3de30f72ee4a753795677afc

The wrappers in this folder re-export `walletforge_x402.langchain_tools`.

## Install

```bash
pip install "git+https://github.com/mig26-design/walletforge-x402.git#egg=walletforge-x402[langchain]"
```

Paid calls require a Base-funded EOA:

```bash
export BUYER_PRIVATE_KEY=0xYOUR_FUNDED_BASE_EOA_KEY
```

Never commit or print that key. Unpaid 402 smoke does not need it and does not spend.

## Unpaid 402 (no spend)

```bash
curl -sS -i -X POST https://api.walletforge.app/v1/fetch-markdown \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://example.com","max_chars":5000}'
```

```bash
curl -sS -i -X POST https://api.walletforge.app/v1/normalize \
  -H 'Content-Type: application/json' \
  -d '{"text":"hello"}'
```

Expected: **HTTP 402**, `PAYMENT-REQUIRED` header, JSON body with
`x402Version: 2`, `accepts[0].network = eip155:8453`, and amounts `50000` /
`10000` (USDC 6 decimals).

## LangChain usage

```python
from examples.community_tools.walletforge_x402 import walletforge_tools

tools = walletforge_tools()  # fetch_markdown, normalize_text
# bind tools to your agent / chat model as usual
```

Equivalent import from the published package:

```python
from walletforge_x402.langchain_tools import walletforge_tools

tools = walletforge_tools()
```

Safe example (unpaid 402 unless you set a key and pass `--invoke`):

```bash
python examples/community_tools/walletforge_x402/langchain_example.py
```

`--invoke` spends about **0.05 USDC** on Base mainnet.

## Mocked tests (no live spend)

```bash
pytest examples/community_tools/walletforge_x402/tests/test_tools_mocked.py
```

Fixtures under `tests/fixtures/` are captured unpaid 402 bodies. Tests mock the
standalone package and `httpx`; they do not settle on-chain.

## Docs listing (correct long-term home)

Please list this integration in [langchain-ai/docs](https://github.com/langchain-ai/docs)
(`scripts/data/integration_external_docs.yaml`), for example:

```yaml
- name: WalletForgeTools
  docs_url: https://github.com/mig26-design/walletforge-x402
```

Do not add a hosted MDX guide unless the package meets the 50k monthly download
or featured-integration bar.
