<a name="readme-top"></a>

<div align="center">
  <h1 align="center" style="border-bottom: none">Orcest AI: Intelligent Search & Research API</h1>
  <p align="center"><b>The Orcest AI Ecosystem Core</b></p>
</div>

<div align="center">
  <a href="https://github.com/orcest-ai/orcest.ai/blob/main/LICENSE"><img src="https://img.shields.io/badge/LICENSE-MIT-20B2AA?style=for-the-badge" alt="MIT License"></a>
</div>

<hr>

Orcest AI provides real-time search, extraction, research, and web crawling through a single, secure API. It is the core of the **Orcest AI** ecosystem, integrated with **RainyModel** (rm.orcest.ai) for intelligent LLM routing.

### Orcest AI Ecosystem

| Service | Domain | Role |
|---------|--------|------|
| **Orcest AI** | orcest.ai | Search & Research API |
| **Lamino** | llm.orcest.ai | LLM Workspace |
| **RainyModel** | rm.orcest.ai | LLM Routing Proxy |
| **Maestrist** | agent.orcest.ai | AI Agent Platform |
| **Orcide** | ide.orcest.ai | Cloud IDE |
| **Login** | login.orcest.ai | SSO Authentication |

## Features

- **Real-time Search**: Intelligent web search and information retrieval
- **Data Extraction**: Structured data extraction from web sources
- **Research API**: Automated research workflows
- **Web Crawling**: Secure, scalable web crawling
- **RainyModel Integration**: Smart LLM routing (Free -> Internal -> Premium)
- **SSO Authentication**: Enterprise-grade access control

## API Endpoints

```
GET  /health    - Health check
POST /search    - Search API
POST /extract   - Data extraction
POST /research  - Research workflows
```

## Deployment

Deployed on Render with auto-deploy from `main` branch.

## Development

This project uses a modular architecture with partner integrations for various LLM providers.

## License

This project is licensed under the [MIT License](LICENSE).

Part of the [Orcest AI](https://orcest.ai) ecosystem.
