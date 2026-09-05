FROM python:3.12-slim

# Install Tesseract OCR (system-level dependency) and Poppler (required by
# pdf2image to render PDF pages into images before OCR)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY . .

# Qdrant local storage will live here — mount a volume to this path
VOLUME ["/app/qdrant_data"]

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]