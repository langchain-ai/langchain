FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the agent code and config (not the full monorepo libs/)
COPY agents/ agents/
COPY .env.example .env.example
COPY migrations/ migrations/

# Default command — can be overridden via Railway service settings
# Runs the SEO agent CLI; Railway will set the specific command
CMD ["python", "-m", "agents.seo_agent.run", "--help"]
