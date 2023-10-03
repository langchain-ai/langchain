FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir .

CMD exec uvicorn ____project_name_identifier.server:app --host 0.0.0.0 --port $PORT

