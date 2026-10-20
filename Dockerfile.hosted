FROM python:3.11-slim

ENV RAG_MODE=hosted \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-hosted.txt ./
RUN pip install --no-cache-dir -r requirements-hosted.txt

COPY app.py ./
COPY core/ ./core/
COPY templates/ ./templates/

EXPOSE 10000
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 2 --timeout 180 app:app"]
