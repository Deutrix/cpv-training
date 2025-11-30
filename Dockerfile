FROM python:3.11

# Workdir
WORKDIR /app

# Copy requirements first (bolji cache)
COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . /app

# Expose port (Coolify će ga očitati)
EXPOSE 8000

# Start FastAPI preko uvicorn-a
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
