# ============================================================================
# Dockerfile for Video Anomaly Detection System (WS-MIL)
# ============================================================================
FROM python:3.10-slim

WORKDIR /app

# Install system runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and configurations
COPY . .

# Create necessary directories
RUN mkdir -p /app/data \
             /app/experiments \
             /app/uploads \
             /app/xd_violence_data

# Expose Streamlit (8501) and FastAPI (8000)
EXPOSE 8000 8501

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command starts Streamlit dashboard
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
