FROM python:3.11-slim

WORKDIR /app

COPY . /app

# TODO install dependencies with pip

CMD exec uvicorn ____project_name_identifier.server:app --host 0.0.0.0 --port $PORT

