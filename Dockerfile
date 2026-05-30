# Dockerfile — ZORO bot + FastAPI (Phase 3)
FROM python:3.11-slim

WORKDIR /app

# System dependencies (for psycopg2)
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY . .

# Make start script executable
RUN chmod +x start.sh

# FastAPI port
EXPOSE 8000

CMD ["./start.sh"]
