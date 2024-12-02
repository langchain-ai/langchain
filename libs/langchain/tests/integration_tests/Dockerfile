FROM python:3.11-slim

RUN apt-get update && apt-get install -y build-essential && \
    pip install --upgrade pip && \
    pip install --no-cache-dir langchain langchain-community langchain-openai langchain-weaviate pytest weaviate && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /root/.cache/pip/ && \
    apt-get remove --purge -y build-essential && \
    apt-get autoremove -y

# Copy the source code
COPY . /app

WORKDIR /app

CMD ["pytest", "-sv", "./retrievers/self_query"]
