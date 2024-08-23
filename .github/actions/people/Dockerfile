FROM python:3.9

RUN pip install httpx PyGithub "pydantic==2.0.2" pydantic-settings "pyyaml>=5.3.1,<6.0.0"

COPY ./app /app

CMD ["python", "/app/main.py"]