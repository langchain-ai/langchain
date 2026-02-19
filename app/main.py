"""
Orcest.ai - The Self-Adaptive LLM Orchestrator
Platform for reliable AI agents
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

app = FastAPI(
    title="Orcest.ai",
    description="The Self-Adaptive LLM Orchestrator platform for reliable AI agents",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RAINYMODEL_BASE_URL = os.getenv("RAINYMODEL_BASE_URL", "https://rm.orcest.ai/v1")

LANDING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Orcest AI - Intelligent LLM Orchestration</title>
<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-2701288361875881"
     crossorigin="anonymous"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0a0a0f;color:#e0e0e0;min-height:100vh}
.hero{text-align:center;padding:80px 20px 40px;background:linear-gradient(135deg,#0a0a1a 0%,#1a1a3a 50%,#0a0a1a 100%)}
.hero h1{font-size:3rem;font-weight:800;background:linear-gradient(135deg,#60a5fa,#a78bfa,#f472b6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:16px}
.hero p{font-size:1.2rem;color:#94a3b8;max-width:600px;margin:0 auto}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:24px;max-width:1100px;margin:40px auto;padding:0 20px}
.card{background:#111827;border:1px solid #1f2937;border-radius:16px;padding:32px;transition:all .3s;text-decoration:none;color:inherit;display:block}
.card:hover{border-color:#3b82f6;transform:translateY(-4px);box-shadow:0 12px 40px rgba(59,130,246,.15)}
.card h3{font-size:1.3rem;color:#f8fafc;margin-bottom:8px}
.card .tag{display:inline-block;font-size:.75rem;padding:3px 10px;border-radius:20px;margin-bottom:12px;font-weight:600}
.tag-api{background:#1e3a5f;color:#60a5fa}
.tag-chat{background:#1e3a1e;color:#4ade80}
.tag-ide{background:#3a1e3a;color:#c084fc}
.tag-agent{background:#3a2e1e;color:#fbbf24}
.tag-llm{background:#1e2a3a;color:#38bdf8}
.tag-sso{background:#2a1e1e;color:#fb923c}
.card p{color:#94a3b8;font-size:.95rem;line-height:1.5}
.card .url{color:#60a5fa;font-size:.85rem;margin-top:12px;display:block}
footer{text-align:center;padding:40px 20px;color:#64748b;border-top:1px solid #1f2937;margin-top:40px}
footer a{color:#60a5fa;text-decoration:none}
footer a:hover{text-decoration:underline}
.badge{display:flex;gap:8px;justify-content:center;margin-top:20px;flex-wrap:wrap}
.badge span{font-size:.8rem;padding:4px 12px;border-radius:20px;background:#1f2937;color:#94a3b8}
</style>
</head>
<body>
<div class="hero">
<h1>Orcest AI</h1>
<p>Intelligent LLM Orchestration Platform &mdash; unified routing across free, internal, and premium AI models.</p>
<div class="badge">
<span>RainyModel Routing</span>
<span>OpenAI Compatible</span>
<span>Multi-Provider</span>
<span>Self-Hosted Options</span>
</div>
</div>
<div class="grid">
<a href="https://rm.orcest.ai" class="card">
<span class="tag tag-llm">LLM Proxy</span>
<h3>RainyModel</h3>
<p>Intelligent LLM routing proxy. Routes requests through free, internal, and premium backends automatically.</p>
<span class="url">rm.orcest.ai</span>
</a>
<a href="https://llm.orcest.ai" class="card">
<span class="tag tag-chat">Chat</span>
<h3>Lamino</h3>
<p>All-in-one AI chat with RAG, document processing, and workspace management. Powered by RainyModel.</p>
<span class="url">llm.orcest.ai</span>
</a>
<a href="https://agent.orcest.ai" class="card">
<span class="tag tag-agent">Agent</span>
<h3>Maestrist</h3>
<p>AI-driven software development agent. Autonomous coding, debugging, and project management.</p>
<span class="url">agent.orcest.ai</span>
</a>
<a href="https://ide.orcest.ai" class="card">
<span class="tag tag-ide">IDE</span>
<h3>Orcide</h3>
<p>AI-powered code editor with intelligent autocomplete, inline chat, and code generation.</p>
<span class="url">ide.orcest.ai</span>
</a>
<a href="/docs" class="card">
<span class="tag tag-api">API</span>
<h3>Orcest API</h3>
<p>Real-time search, extraction, research, and web crawling through a single, secure API.</p>
<span class="url">orcest.ai/docs</span>
</a>
<a href="https://login.orcest.ai" class="card">
<span class="tag tag-sso">SSO</span>
<h3>Login Portal</h3>
<p>Single sign-on for all Orcest AI services. Manage your account and API keys.</p>
<span class="url">login.orcest.ai</span>
</a>
</div>
<footer>
<p>Orcest AI &copy; 2025 &mdash; Support: <a href="mailto:admin@danial.ai">admin@danial.ai</a></p>
</footer>
</body>
</html>"""


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "orcest.ai"}


@app.get("/", response_class=HTMLResponse)
async def landing_page():
    return HTMLResponse(content=LANDING_HTML)


@app.get("/api/info")
async def api_info():
    return {
        "platform": "orcest.ai",
        "description": "Real-time search, extraction, research, and web crawling through a single, secure API",
        "services": {
            "rainymodel": "https://rm.orcest.ai",
            "lamino": "https://llm.orcest.ai",
            "maestrist": "https://agent.orcest.ai",
            "orcide": "https://ide.orcest.ai",
            "login": "https://login.orcest.ai",
        },
        "rainymodel_api": RAINYMODEL_BASE_URL,
        "support": "admin@danial.ai",
    }
