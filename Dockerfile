# Use a slim Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies + huggingface-hub
RUN pip install --no-cache-dir -r requirements.txt huggingface-hub

# Copy application code (NOT models - they'll be downloaded from HuggingFace)
COPY main.py .
COPY predictor.py .
COPY cpv_codes.csv .
COPY cpv_embeddings.pt .
COPY static/ ./static/

# Download model from Hugging Face Hub
# Set these as build arguments in Coolify:
# HF_MODEL_ID=Deutrix/cpv-decoder
# HF_TOKEN=hf_xxxxx (your HuggingFace token)
ARG HF_MODEL_ID=Deutrix/cpv-decoder
ARG HF_TOKEN

# Download the model at build time
RUN if [ -n "$HF_TOKEN" ]; then \
    mkdir -p models && \
    huggingface-cli login --token $HF_TOKEN && \
    huggingface-cli download $HF_MODEL_ID --local-dir ./models/cpv-decoder; \
    else \
    echo "ERROR: HF_TOKEN not set. Cannot download private model."; \
    exit 1; \
    fi

# Create a non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API with production settings
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
