FROM python:3.12-slim

# Cache bust: 2026-03-25-v2
WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the agent code and config (not the full monorepo libs/)
COPY agents/ agents/
COPY .env.example .env.example
COPY migrations/ migrations/

# Run the Telegram bot as the main process
CMD ["python", "-m", "agents.seo_agent.telegram_bot"]
