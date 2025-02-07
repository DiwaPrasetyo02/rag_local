FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY src/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src/ .
RUN mkdir -p data models

EXPOSE 8000

CMD ["uvicorn", "rag_app.main:app", "--host", "0.0.0.0", "--port", "8000"]