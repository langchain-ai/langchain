FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e . fastapi uvicorn[standard] langchain ollama agent-zero
ENV PERSIST_DIR=/data/db
CMD ["uvicorn", "examples.rag_webapp.app:app", "--host", "0.0.0.0", "--port", "8000"]
