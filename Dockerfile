FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    stockfish \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code (model file is NOT copied — it's downloaded at startup)
COPY . .

EXPOSE 8000

# Run download script first, then start the API
CMD python download_model.py && uvicorn main:app --host 0.0.0.0 --port 8000
