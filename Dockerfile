FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --upgrade pip && pip install .

COPY main.py ./main.py

EXPOSE 8080
CMD ["python", "main.py"]
