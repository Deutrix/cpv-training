# Deploying CPV Decoder with Large Model Files

## Problem
Your model files (~1.68 GB) are too large for standard Git push. GitHub has limits on file sizes and push timeouts.

## Solutions

### Option 1: Git LFS (Recommended for GitHub)

**Install Git LFS:**
```bash
# Download from: https://git-lfs.github.com/
# Or install via package manager
winget install GitHub.GitLFS
# or
choco install git-lfs
```

**Setup Git LFS in your repository:**
```bash
cd c:\Users\boris.AORUS\Desktop\cpv-decoder

# Initialize Git LFS
git lfs install

# Track large model files
git lfs track "models/**/*.bin"
git lfs track "models/**/*.safetensors"
git lfs track "*.pt"
git lfs track "*.pth"

# Add .gitattributes (created by git lfs track)
git add .gitattributes

# Add your files
git add .
git commit -m "Add model files with Git LFS"

# Push (LFS will handle large files)
git push origin main
```

**Note:** GitHub LFS has storage limits:
- Free: 1 GB storage, 1 GB bandwidth/month
- You may need to upgrade or use alternative hosting

---

### Option 2: External Model Storage (Recommended for Production)

Store models externally and download them at container startup.

#### A. Using Hugging Face Hub (Free & Easy)

**1. Upload model to Hugging Face:**
```bash
# Install huggingface-hub
pip install huggingface-hub

# Login
huggingface-cli login

# Upload your model
huggingface-cli upload your-username/cpv-decoder ./models/cpv-decoder
```

**2. Update Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies including huggingface-hub
RUN apt-get update && apt-get install -y --no-install-recommends curl git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt huggingface-hub

# Copy application code (NOT models)
COPY main.py predictor.py cpv_codes.csv cpv_embeddings.pt ./
COPY static/ ./static/

# Download model at build time
ARG HF_MODEL_ID=your-username/cpv-decoder
RUN mkdir -p models && \
    huggingface-cli download ${HF_MODEL_ID} --local-dir ./models/cpv-decoder

# Security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

**3. In Coolify, set environment variable:**
```
HF_MODEL_ID=your-username/cpv-decoder
```

#### B. Using Cloud Storage (S3, Google Cloud Storage, etc.)

**1. Upload model to S3:**
```bash
aws s3 cp models/cpv-decoder/ s3://your-bucket/cpv-decoder/ --recursive
```

**2. Update Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl awscli && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py predictor.py cpv_codes.csv cpv_embeddings.pt ./
COPY static/ ./static/

# Download model from S3
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG MODEL_S3_PATH=s3://your-bucket/cpv-decoder/

RUN mkdir -p models/cpv-decoder && \
    aws s3 cp ${MODEL_S3_PATH} ./models/cpv-decoder/ --recursive

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

---

### Option 3: Direct Coolify Deployment (No Git)

Deploy directly to Coolify without Git:

**1. Build Docker image locally:**
```bash
cd c:\Users\boris.AORUS\Desktop\cpv-decoder
docker build -t cpv-decoder:latest .
```

**2. Push to Docker Hub:**
```bash
docker login
docker tag cpv-decoder:latest your-dockerhub-username/cpv-decoder:latest
docker push your-dockerhub-username/cpv-decoder:latest
```

**3. In Coolify:**
- Create new application
- Select "Docker Image" instead of Git
- Use image: `your-dockerhub-username/cpv-decoder:latest`
- Configure port 8000
- Deploy

---

### Option 4: Split Repository (Quick Fix)

Keep model files separate from code:

**1. Create .gitignore:**
```bash
# Add to .gitignore
models/
*.pt
*.pth
```

**2. Push code only:**
```bash
git add .
git commit -m "Add application code (without models)"
git push origin main
```

**3. Upload models separately:**
- Use Git LFS (Option 1)
- Or external storage (Option 2)
- Or include in Docker image and push to Docker Hub (Option 3)

---

## Recommended Approach

**For Coolify Deployment:**

1. **Use Hugging Face Hub** (Option 2A) - Free, easy, designed for ML models
2. **Modify Dockerfile** to download model at build time
3. **Push code to GitHub** (without models)
4. **Deploy to Coolify** from GitHub

**Advantages:**
- ✅ No Git LFS quota issues
- ✅ Fast code updates (no large files)
- ✅ Model versioning on Hugging Face
- ✅ Free hosting for models
- ✅ Works seamlessly with Coolify

---

## Quick Start: Hugging Face Method

```bash
# 1. Install huggingface-hub
pip install huggingface-hub

# 2. Login to Hugging Face
huggingface-cli login

# 3. Upload model
huggingface-cli upload your-username/cpv-decoder ./models/cpv-decoder

# 4. Add models/ to .gitignore
echo "models/" >> .gitignore

# 5. Push code to GitHub
git add .
git commit -m "Prepare for Coolify deployment with external model storage"
git push origin main

# 6. Deploy to Coolify with HF_MODEL_ID environment variable
```

---

## Comparison

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| Git LFS | Simple, integrated | Quota limits, costs | Small teams, paid plans |
| Hugging Face | Free, ML-focused | Extra step | ML models, open source |
| S3/Cloud | Scalable, reliable | Costs, complexity | Production, large scale |
| Docker Hub | No Git needed | Large image size | Quick deployments |
| Split Repo | Simple | Manual sync | Development only |

---

## Need Help?

Choose the method that fits your needs:
- **Quick & Free:** Hugging Face Hub
- **Enterprise:** S3/Cloud Storage
- **Simple:** Docker Hub direct push
- **GitHub-only:** Git LFS (with paid plan)
